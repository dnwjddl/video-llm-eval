# oracle — 최적 비주얼 토큰 부분집합 측정

**질문**: 규칙(attention·유사도) 없이, "전체 토큰을 넣었을 때와 같은 답이 나오는 최소 토큰 집합"을
비디오·질문마다 직접 찾으면 어떤 모양이고, 기존 압축 방법은 그것과 얼마나 다른가?

**방법**: LLaVA-OneVision(frozen)의 비주얼 토큰(196/frame × 32 = 6,272개)마다 마스크 m_i 를 두고,
LLM attention logit 에 `log m_i` 를 더해 가린다. 손실 `KL(p_full ‖ p_m) + λ·Σm` 을 m 에 대해서만
gradient 로 최적화하고, λ 를 훑어 (토큰 수, KL) 곡선을 얻는다. 모델 가중치는 학습하지 않는다.

## 파일

| 파일 | 역할 |
|---|---|
| `llava_hooks.py` | 모델 로드, 입력 임베딩 + visual span 추출, attention bias 패치, 삭제 forward, 답 분포 비교 |
| `stage0_equivalence.py` | **0단계**: bias 마스킹 ≈ 실제 삭제 등가성 검증, position 관례 비교, fwd+bwd 비용 측정 |
| `stage1_mask_opt.py` | **1단계**: 비디오(·질문)별 λ sweep 마스크 최적화 → (\|S\|, KL) 곡선 + 실제 삭제 재검증 + 기준선 |
| `plot_stage1.py` | 1단계 결과 집계·그림 (GPU 불필요): RD 곡선, b*(ε) 분포와 기준선 배수, task_type 별 b*, 답 보존 곡선, 마스크 시간·공간 프로파일, λ 간 포함 비율, 디렉터리 간 비교 |

## 0단계 실행

```bash
conda activate llava
# 파이프라인 확인 (0.5B, 5 샘플, 몇 분)
python oracle/stage0_equivalence.py --pretrained lmms-lab/llava-onevision-qwen2-0.5b-ov --n 5 --timing
# 본 검증 (7B, 20 샘플)
python oracle/stage0_equivalence.py --pretrained lmms-lab/llava-onevision-qwen2-7b-ov --n 20 --timing
```

비디오는 `latency/extract_videos_subset.py` 로 풀어둔 Video-MME 150개(`~/videomme_videos`)를 재사용한다.

### 무엇을 보나

1. **패치 무해성**: bias 를 전부 0 으로 두고 돌린 logit 이 패치 없는 full 과 같아야 한다.
2. **(1) bias vs 삭제(position 유지)**: 기본값 `--kernel math` 에서는 모든 경로가 같은 SDPA 커널을 쓰므로
   이 차이는 이론상 0 에 가까워야 한다 (KL ≪ 잡음 바닥, argmax 전부 일치). 함께 찍히는 "잡음 바닥"
   (같은 full 입력을 기본 커널 vs math 커널로 돌린 차이)이 bf16 커널 차이의 크기다. `--kernel default` 로
   돌리면 (1)에 그 커널 차이가 섞여 들어오므로 잡음 바닥과 같은 자릿수면 통과로 본다.
3. **(2) bias vs 삭제(renumber)**: LLaVA-OV 는 1D RoPE 라 토큰을 지우면 뒤 토큰 위치가 밀린다.
   (2)가 (1)보다 뚜렷이 크면 "position 유지"를 기본 관례로 채택하고, 논문에서는 둘 다 보고.
3-1. **fp32 확정 실험**: `--dtype float32` (0.5B) 로 돌리면 (1)이 ~1e-6 수준으로 떨어져야 한다. 그러면 bf16 에서
   남는 1e-3 KL 은 softmax 누적 반올림(가린 경로는 6,380 key 위에서, 삭제 경로는 짧은 시퀀스에서 정규화)임이 확정된다.
3-2. **keydim 구현 vs bias 구현**: 기본 마스크 구현은 `keydim` (q/k 에 차원을 덧붙여 q_extra·k_extra = log m_j).
   mask 가 필요 없어 is_causal 경로(flash)를 그대로 쓰고 gradient 는 k 로 흐른다. (1)은 keydim, (1')은 bias 구현이며
   둘 다 삭제와 같아야 하고, 둘 사이 차이는 잡음 바닥 수준이어야 한다.
4. **(3) fwd+bwd 비용**: gradient 경로는 `--grad_kernel auto` 가 기본. memory-efficient SDPA 를 먼저 시도하고
   (구버전 PyTorch 의 `LSE is not correctly aligned` 오류를 피하려고 시퀀스를 `--pad_multiple 64` 로 padding),
   실패하면 math 로 떨어진다. math 는 6.4k×6.4k attention 을 실제로 만들어 0.5B 에서도 2.6 s/step 이라
   1단계 규모에 맞지 않으므로, efficient 커널이 살아나는지가 중요하다. 7B 기준 step 시간과 peak 메모리. `grad|θ|` 가 0 이면 gradient 가 마스크까지
   안 흐르는 것 (non-reentrant checkpointing 이 켜졌는지 확인). sdpa 에서 backward 가 실패하면
   `--attn_impl eager` 로 재시도.

결과는 `oracle/results/stage0_<model>.json` 에 저장된다.

## 구현 메모

- 마스킹은 `Qwen2SdpaAttention.forward` 를 감싸서 4D attention mask 에 key 축 bias 를 더한다.
  HF sdpa 경로는 패딩이 없으면 mask 를 `None` 으로 넘기므로 그 경우 causal mask 를 직접 만든다.
- 비주얼 구간은 `prepare_inputs_labels_for_multimodal` 의 출력 길이에서 텍스트 토큰 수를 빼서 구한다.
  LLaVA-OV 는 `mm_newline_position=one_token` 이라 구간 끝에 `image_newline` 1개가 붙고, 이건 항상 유지한다.
- 마스크 단위는 `get_2dPool` 이후 토큰(프레임당 196 = 2×2 patch 묶음)이다. pooling 이전에 걸면
  pooling 이 가린 patch 와 안 가린 patch 를 섞어 "지운다"의 의미가 깨진다.
- transformers 4.40 의 Qwen2 는 RoPE cos/sin 테이블을 kv_seq_len 까지만 잘라 `cos[position_ids]` 로 인덱싱한다.
  "삭제 후 position 유지"는 position id 가 시퀀스 길이보다 크므로 그대로 두면 CUDA index-out-of-bounds 가 난다.
  패치가 각 layer 의 `rotary_emb.forward` 를 감싸 테이블을 `max(position)+1` 까지 만들게 한다.
- flash-attn 은 임의 bias 를 받지 못하므로 sdpa 또는 eager 만 사용한다.

## 1단계 실행

```bash
# 파이프라인 확인 (0.5B, 비디오 2개)
python oracle/stage1_mask_opt.py --pretrained lmms-lab/llava-onevision-qwen2-0.5b-ov --mode agnostic --limit 2
# 파일럿 (7B, 비디오 100개, 질문 모름 = 비디오당 마스크 하나)
python oracle/stage1_mask_opt.py --pretrained lmms-lab/llava-onevision-qwen2-7b-ov --mode agnostic --limit 100 --resume
# 질문별 마스크 (같은 비디오, 비교용)
python oracle/stage1_mask_opt.py --pretrained lmms-lab/llava-onevision-qwen2-7b-ov --mode aware --limit 100 --resume
```

- 데이터: Video-MME 로컬 비디오 145개 × 그 비디오의 질문 전부(3개, `task_type` 라벨). 새로 받을 것 없음.
- 비디오마다 λ 6개(10배 간격 1e-3~100) × (150 + 5×100) = 650 step. λ 당 50 step 은 keep 이 계속 줄고 있는 채로 끝나 곡선 점이 궤적 스냅샷이 되므로 100 으로. (0.5B 파일럿: λ=0.03 에서 이미 keep 6%, 51토큰(0.8%)에서도 답 유지 → λ 를 1e-3 부터.) 기본 `--q_per_step 1` 은 step 마다 질문 하나를 돌아가며 써서
  agnostic 의 비용을 aware 와 같게 맞춘다. 0단계 timing 의 s/step × 450 이 비디오당 최적화 시간.
- 출력 `results/stage1_<model>_<mode>/<videoID>.json`: λ 점마다 `n_keep`, oracle(실제 삭제) KL·full 과 답 일치율·정확도,
  같은 개수의 random / frame_uniform / grid 기준선 KL. `masks_<videoID>.npz` 에 soft 마스크와 0/1 마스크.
- `--resume` 로 중단 지점부터 이어서. 같은 결과 디렉터리에 GPU 를 더 붙이려면 새 프로세스에 `--reverse` 를 주면
  뒤에서부터 처리해 중간에서 만난다. 작업 중인 비디오는 `<vid>.json.lock` 으로 표시되어 다른 프로세스가 건너뛴다
  (프로세스가 죽어 락이 남으면 `--lock_ttl` 3시간 뒤 무시됨, 또는 수동 삭제).
- **`--verifier`**: `letters`(기본) 는 객관식 글자 분포 KL — 정보량이 2 bit 라 0.5B 에서 51 토큰(0.8%)으로도 답이 유지됐다.
  `caption` 은 전체 토큰으로 생성한 비디오 설명(`--caption_tokens 96`)을 teacher-forcing 한 토큰별 full-vocab KL —
  "모델의 이해가 보존되는 최소 subset" 에 가깝고 질문 라벨이 필요 없다. agnostic 에 권장. `both` 는 둘 다.
  결과 디렉터리 이름에 verifier 가 붙는다 (`stage1_<model>_<mode>_<verifier>`).
  0.5B 파일럿: letters oracle 은 5% 에서 답 유지, caption oracle 은 KL 0.0015 에 61~66%, KL 0.04 에 15~21% 필요 —
  "답에 필요한 양" 과 "이해 보존에 필요한 양" 의 간격. `cap_agree` 는 full-token argmax 와의 일치율, `gen_agree` 는
  생성 캡션이 raw argmax 와 같은지(repetition_penalty 를 꺼서 ≈1.0 이어야).

## 1단계 집계·그림

```bash
python oracle/plot_stage1.py oracle/results/stage1_llava-onevision-qwen2-7b-ov_agnostic_letters
python oracle/plot_stage1.py oracle/results/stage1_..._agnostic_caption --compare oracle/results/stage1_..._agnostic_letters --labels caption letters
```

`<dir>/figs/` 에 그림 6장, `<dir>/summary.md` 에 표 (λ 점별 중앙값, ε 별 최소 충분 예산 b*(ε) 와 기준선 배수,
task_type 별 b*, 마스크 프로파일, λ 간 포함 비율). 진행 중인 결과 디렉터리에도 언제든 돌릴 수 있다.
b*(ε) 는 (keep, KL) 곡선을 log-log 보간해 KL ≤ ε 인 최소 keep 으로 정의하며, 곡선의 최소점도 ε 이하면 ↓, 최대점도
ε 초과면 ↑(=1.0 처리) 로 표시한다. λ 간 포함 비율은 warm start 때문에 낙관적이므로 nested 검정은 cold start 로 따로 한다.

## 다음 단계 (예정)

- attention top-k / 유사도 병합 기준선 추가. nested 검정(λ 간 S 포함 관계), seed 안정성.
