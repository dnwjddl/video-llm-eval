"""LLaVA-OneVision(Qwen2 LLM) 위에서 "비주얼 토큰 마스킹"을 구현하는 공용 훅.

핵심 아이디어
  - 비디오는 encoder(SigLIP) → get_2dPool(196/frame) → MLP projector 를 거쳐
    LLM 입력 시퀀스의 연속 구간(visual span)에 삽입된다.
  - 마스킹은 LLM attention logit에 key 별 additive bias 를 더하는 것으로 구현한다.
      bias_j = log m_j  (m_j ∈ (0,1]) ; m_j = 0 이면 -inf → 그 토큰은 아무도 보지 못함.
    이는 토큰을 실제로 지우는 것과 "position id 를 유지한 삭제"에 대해 등가여야 한다
    (stage0_equivalence.py 가 이를 검증).
  - 모델 가중치는 절대 건드리지 않는다. 학습되는 것은 (나중 단계에서) bias 뿐이다.

환경: conda `llava` (transformers==4.40.0, LLaVA-NeXT). attn_implementation 은
sdpa 또는 eager 를 지원한다 (flash-attn 은 임의 bias 를 못 받으므로 사용 불가).
"""

import re
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F

LETTERS = "ABCDE"


# ----------------------------------------------------------------------------
# 모델 로드 / 입력 구성
# ----------------------------------------------------------------------------
def load_llava(pretrained: str, attn_impl: str = "sdpa"):
    """LLaVA-OneVision 로드 (bf16, 단일 GPU). 모든 파라미터 requires_grad=False."""
    from llava.model.builder import load_pretrained_model

    tokenizer, model, image_processor, _ = load_pretrained_model(
        pretrained, None, "llava_qwen", device_map="cuda",
        torch_dtype="bfloat16", attn_implementation=attn_impl,
    )
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tokenizer, model, image_processor


def build_prompt_ids(tokenizer, question: str) -> torch.Tensor:
    """qwen_1_5 대화 템플릿으로 <image>\\n{question} 프롬프트를 토큰화. (T,) LongTensor."""
    from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
    from llava.conversation import conv_templates
    from llava.mm_utils import tokenizer_image_token

    conv = conv_templates["qwen_1_5"].copy()
    conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
    conv.append_message(conv.roles[1], None)
    return tokenizer_image_token(conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt")


@dataclass
class VideoInputs:
    embeds: torch.Tensor      # (L, d)  LLM 입력 임베딩 (텍스트 + 비주얼 + newline)
    vis_start: int            # visual span 시작 index
    n_vis: int                # 마스킹 대상 비주얼 토큰 수 (frames * 196)
    has_newline: bool         # span 끝에 image_newline 토큰이 붙어 있는지
    n_frames: int

    @property
    def vis_end(self) -> int:  # 마스킹 대상 구간 [vis_start, vis_end)
        return self.vis_start + self.n_vis

    @property
    def L(self) -> int:
        return self.embeds.shape[0]

    def vis_index(self) -> torch.Tensor:
        return torch.arange(self.vis_start, self.vis_end, device=self.embeds.device)


@torch.no_grad()
def encode_video_inputs(model, image_processor, frames, input_ids: torch.Tensor) -> VideoInputs:
    """frames (T,H,W,3 uint8) + 프롬프트 → LLM 입력 임베딩과 visual span.

    llava 의 prepare_inputs_labels_for_multimodal 을 그대로 호출하므로 pooling(196/frame),
    projector, newline 처리가 lmms-eval 평가와 완전히 동일하다.
    """
    from llava.constants import IMAGE_TOKEN_INDEX

    video = image_processor.preprocess(frames, return_tensors="pt")["pixel_values"]
    video = video.to(device=model.device, dtype=torch.bfloat16)
    ids = input_ids.unsqueeze(0).to(model.device)
    _, _, _, _, embeds, _ = model.prepare_inputs_labels_for_multimodal(
        ids, None, None, None, None, [video], ["video"])
    embeds = embeds[0]  # (L, d)

    img_pos = (input_ids == IMAGE_TOKEN_INDEX).nonzero().flatten()
    assert img_pos.numel() == 1, "프롬프트에 <image> 토큰이 정확히 하나여야 합니다"
    vis_start = int(img_pos[0])
    n_text = input_ids.numel() - 1
    span = embeds.shape[0] - n_text          # visual span 길이 (newline 포함 가능)

    cfg = model.config
    has_newline = ("unpad" in getattr(cfg, "mm_patch_merge_type", "")
                   and getattr(cfg, "mm_newline_position", "one_token") == "one_token")
    n_frames = int(video.shape[0])
    n_vis = span - (1 if has_newline else 0)
    assert n_vis % n_frames == 0, f"visual span {span} 이 프레임 수 {n_frames} 로 나뉘지 않음 (newline 판정 확인)"
    return VideoInputs(embeds=embeds, vis_start=vis_start, n_vis=n_vis, has_newline=has_newline, n_frames=n_frames)


# ----------------------------------------------------------------------------
# Attention bias 패치
# ----------------------------------------------------------------------------
class _BiasState:
    bias: Optional[torch.Tensor] = None   # (1,1,1,L) — key 축 additive bias, 모든 query 공통
    rope_len: Optional[int] = None        # position_ids.max()+1 — RoPE 테이블을 이 길이까지 만들게 함


_STATE = _BiasState()
_PATCHED = {}


def install_bias_patch(model):
    """모델이 실제로 쓰는 attention 클래스 하나만 패치한다 (중복 적용 방지).

    두 가지를 처리한다.
    (a) visual bias: 4D attention mask 에 key 축 bias 를 더한다.
    (b) RoPE 길이: transformers 4.40 의 Qwen2 는 cos/sin 테이블을 kv_seq_len 까지만 잘라서
        cos[position_ids] 로 인덱싱한다. "삭제 후 position 유지" 처럼 position id 가 시퀀스
        길이보다 크면 index out of bounds 가 나므로, 테이블을 max(position)+1 까지 만들게 한다.
    """
    attn_cls = type(model.model.layers[0].self_attn)
    if attn_cls in _PATCHED:
        return attn_cls
    orig = attn_cls.forward

    def forward(self, hidden_states, attention_mask=None, position_ids=None,
                past_key_value=None, output_attentions=False, use_cache=False, **kw):
        b = _STATE.bias
        if b is not None:
            bsz, q_len, _ = hidden_states.shape
            if attention_mask is None:
                # HF sdpa 경로는 패딩이 없으면 4D mask 를 None 으로 넘기고 is_causal=True 를 쓴다.
                # bias 를 더하려면 causal mask 를 직접 만들어야 한다.
                mn = torch.finfo(hidden_states.dtype).min
                causal = torch.full((q_len, q_len), mn, device=hidden_states.device,
                                    dtype=hidden_states.dtype).triu(1)
                attention_mask = causal[None, None].expand(bsz, 1, q_len, q_len)
            kv = attention_mask.shape[-1]
            attention_mask = attention_mask + b[..., :kv].to(attention_mask.dtype)

        rope = self.rotary_emb
        need = _STATE.rope_len
        if need is None and position_ids is not None:
            # (gradient checkpointing 재계산처럼 _STATE 가 비어 있을 때도 안전하도록 직접 계산)
            need = int(position_ids.max()) + 1
        if need is not None and need > hidden_states.shape[1]:
            orig_rot = rope.forward

            def rot_forward(x, seq_len=None, _orig=orig_rot, _need=need):
                return _orig(x, seq_len=max(seq_len or 0, _need))

            rope.forward = rot_forward
        try:
            return orig(self, hidden_states, attention_mask=attention_mask, position_ids=position_ids,
                        past_key_value=past_key_value, output_attentions=output_attentions,
                        use_cache=use_cache, **kw)
        finally:
            if need is not None and need > hidden_states.shape[1]:
                del rope.forward   # 인스턴스 속성 제거 → 클래스 forward 로 복귀

    attn_cls.forward = forward
    _PATCHED[attn_cls] = orig
    return attn_cls


def make_bias(vi: VideoInputs, m: torch.Tensor) -> torch.Tensor:
    """m: (n_vis,) ∈ [0,1] 마스크 → (1,1,1,L) float32 bias. 비주얼 구간 밖은 0."""
    m = m.float()
    logm = torch.log(m.clamp_min(1e-30))
    # m == 0 인 토큰은 완전히 가린다 (bf16 최소값; causal mask 값과 더해져 -inf 가 되어도 무방)
    logm = torch.where(m > 0, logm, torch.full_like(logm, torch.finfo(torch.bfloat16).min))
    bias = torch.zeros(vi.L, device=vi.embeds.device, dtype=torch.float32)
    bias = bias.index_add(0, vi.vis_index(), logm)   # index_add 는 m 으로 grad 전파 가능
    return bias[None, None, None]


# ----------------------------------------------------------------------------
# Forward 유틸
# ----------------------------------------------------------------------------
def clear_state():
    _STATE.bias = None
    _STATE.rope_len = None


def last_logits(model, embeds: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                bias: Optional[torch.Tensor] = None, clear: bool = True) -> torch.Tensor:
    """마지막 위치의 vocab logits (float32, (vocab,)). bias 가 있으면 그 forward 동안 적용.

    clear=False 이면 forward 후에도 state 를 남긴다. gradient checkpointing 은 backward 때
    layer forward 를 재실행하므로, 그때도 같은 bias 가 보여야 한다 → backward 후 clear_state() 호출.
    """
    _STATE.bias = bias
    _STATE.rope_len = None if position_ids is None else int(position_ids.max()) + 1
    try:
        out = model.model(
            inputs_embeds=embeds[None],
            position_ids=None if position_ids is None else position_ids[None],
            use_cache=False, return_dict=True,
        )
        h = out.last_hidden_state[:, -1]
        return model.lm_head(h)[0].float()
    finally:
        if clear:
            clear_state()


class math_sdpa:
    """SDPA 를 math 백엔드로 강제하는 컨텍스트 — bf16 커널 차이(잡음 바닥) 측정용."""

    def __enter__(self):
        try:
            from torch.nn.attention import SDPBackend, sdpa_kernel
            self._cm = sdpa_kernel(SDPBackend.MATH)
        except Exception:
            self._cm = torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
        return self._cm.__enter__()

    def __exit__(self, *a):
        return self._cm.__exit__(*a)


def delete_tokens(vi: VideoInputs, keep: torch.Tensor, renumber: bool):
    """keep: (n_vis,) bool. 비주얼 토큰 중 keep=False 인 행을 실제로 제거한 (embeds, position_ids)."""
    L = vi.L
    keep_all = torch.ones(L, dtype=torch.bool, device=vi.embeds.device)
    keep_all[vi.vis_start:vi.vis_end] = keep.to(vi.embeds.device)
    idx = keep_all.nonzero().flatten()
    emb = vi.embeds[idx]
    pos = torch.arange(emb.shape[0], device=emb.device) if renumber else idx
    return emb, pos


# ----------------------------------------------------------------------------
# 답 분포 / 비교 지표
# ----------------------------------------------------------------------------
def letters_in_prompt(prompt: str) -> List[str]:
    found = []
    for line in prompt.splitlines():
        m = re.match(r"^\s*\(?([A-E])[.)\]:]", line)
        if m and m.group(1) not in found:
            found.append(m.group(1))
    return found or list("ABCD")


def letter_token_ids(tokenizer, letters: List[str]) -> torch.Tensor:
    ids = []
    for ch in letters:
        t = tokenizer.encode(ch, add_special_tokens=False)
        assert len(t) == 1, f"letter {ch!r} 가 단일 토큰이 아님: {t}"
        ids.append(t[0])
    return torch.tensor(ids)


def letter_dist(logits: torch.Tensor, ids: torch.Tensor) -> torch.Tensor:
    return F.softmax(logits[ids.to(logits.device)], dim=-1)


def compare(logits_a: torch.Tensor, logits_b: torch.Tensor, ids: torch.Tensor) -> dict:
    """a 를 기준으로 b 를 비교. KL(a‖b) on letters, argmax 일치, 전체 vocab logit 최대 차이."""
    pa, pb = letter_dist(logits_a, ids), letter_dist(logits_b, ids)
    kl = float((pa * (pa.clamp_min(1e-12).log() - pb.clamp_min(1e-12).log())).sum())
    return {
        "kl_letters": kl,
        "same_argmax": bool(pa.argmax() == pb.argmax()),
        "max_abs_dlogit_vocab": float((logits_a - logits_b).abs().max()),
        "max_abs_dlogit_letters": float((logits_a[ids] - logits_b[ids]).abs().max()),
    }
