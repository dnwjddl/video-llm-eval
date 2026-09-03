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
4. **(3) fwd+bwd 비용**: gradient 경로는 기본 `--grad_kernel math` 로 돈다. memory-efficient SDPA 는 float mask 의
   backward 에서 `LSE is not correctly aligned (strideH)` 정렬 오류를 낸다 (L=6380 이 8 의 배수가 아님). 7B 기준 step 시간과 peak 메모리. `grad|θ|` 가 0 이면 gradient 가 마스크까지
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
- 비디오마다 λ 6개 × (150 + 5×60) step. 기본 `--q_per_step 1` 은 step 마다 질문 하나를 돌아가며 써서
  agnostic 의 비용을 aware 와 같게 맞춘다. 0단계 timing 의 s/step × 450 이 비디오당 최적화 시간.
- 출력 `results/stage1_<model>_<mode>/<videoID>.json`: λ 점마다 `n_keep`, oracle(실제 삭제) KL·full 과 답 일치율·정확도,
  같은 개수의 random / frame_uniform / grid 기준선 KL. `masks_<videoID>.npz` 에 soft 마스크와 0/1 마스크.
- `--resume` 로 중단 지점부터 이어서.

## 다음 단계 (예정)

- 곡선 집계·그림: 비디오별 (|S|, KL) 곡선, task_type 별 최소 충분 예산, oracle vs 기준선 간격, aware vs agnostic 간격.
- attention top-k / 유사도 병합 기준선 추가. nested 검정(λ 간 S 포함 관계), seed 안정성.
