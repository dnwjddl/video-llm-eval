# video-llm-eval

오픈소스 Video LLM들을 주요 비디오 벤치마크에서 한 번에 검증(evaluate)하기 위한 레포입니다.
평가 하니스로는 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)을 사용합니다 — 아래의 모든 모델과 벤치마크를 공식 지원하므로, 이 레포는 **환경 세팅 + 실행 스크립트 + Gemma 4 플러그인 + 상세 가이드**를 제공합니다.

## 지원 모델

| 모델 | HuggingFace 체크포인트 | lmms-eval 모델 이름 | 비고 |
|---|---|---|---|
| LLaVA-OneVision | `lmms-lab/llava-onevision-qwen2-7b-ov` (0.5B/7B/72B) | `llava_onevision` | LLaVA-NeXT 패키지 필요 |
| LLaVA-Video | `lmms-lab/LLaVA-Video-7B-Qwen2` (7B/72B) | `llava_vid` | LLaVA-NeXT 패키지 필요 |
| Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` (2B/7B/72B) | `qwen2_vl` | `qwen-vl-utils` 필요 |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` (3B/7B/32B/72B) | `qwen2_5_vl` | `qwen-vl-utils` 필요 |
| Gemma 4 | `google/gemma-4-E2B-it`, `google/gemma-4-E4B-it` | `gemma4` | **이 레포의 플러그인으로 지원** ([아래 참고](#gemma-4-플러그인)) |

## 지원 벤치마크

| 벤치마크 | lmms-eval task 이름 | 데이터셋 (HF Hub) | 채점 방식 |
|---|---|---|---|
| Video-MME (w/o subtitle) | `videomme` | `lmms-eval/Video-MME` | 객관식(MCQ), 규칙 기반 |
| Video-MME (w/ subtitle) | `videomme_w_subtitle` | `lmms-eval/Video-MME` | 객관식, 규칙 기반 |
| Video-MME v2 | `videomme_v2` | `MME-Benchmarks/Video-MME-v2` | 그룹 비선형 채점 |
| Video-MME v2 (w/ subtitle) | `videomme_v2_w_subtitle` | `MME-Benchmarks/Video-MME-v2` | 그룹 비선형 채점 |
| MVBench | `mvbench` | `OpenGVLab/MVBench` | 객관식, 규칙 기반 |
| LongVideoBench | `longvideobench_val_v` | `longvideobench/LongVideoBench` | 객관식 (val split) |
| LVBench | `lvbench` | `lmms-eval/LVBench` | 객관식 |
| MLVU | `mlvu_dev` | `sy1998/MLVU_dev` | 객관식 (dev split) |

모든 벤치마크가 객관식/규칙 기반 채점이라 **GPT API 키 없이** 평가 가능합니다.
(`videomme_v2_reasoning` 같은 서브태스크는 별도 judge가 필요할 수 있어 기본 스크립트에서 제외했습니다.)

---

## 1. 설치

```bash
git clone https://github.com/dnwjddl/video-llm-eval.git
cd video-llm-eval
bash setup.sh
```

`setup.sh`가 하는 일:

1. `lmms-eval`을 GitHub main 브랜치에서 설치 (Video-MME v2, Gemma 4 플러그인 지원에 필요)
2. 비디오 디코딩/모델 의존성 설치: `decord`, `qwen-vl-utils`, `torch`, `accelerate` 등
3. LLaVA 계열용 `llava` 패키지 (LLaVA-NeXT) 설치
4. Gemma 4 플러그인 (`gemma4_plugin/`) 설치

> **⚠️ LLaVA 계열은 별도 환경이 필요합니다.** LLaVA-NeXT는 구버전 transformers(`apply_chunking_to_forward` 등 제거된 API)를 기대하는 반면 Gemma 4는 최신 transformers가 필요해서 **한 환경에 공존할 수 없습니다.** LLaVA 전용 환경을 이렇게 만드세요:
>
> ```bash
> conda create -n llava python=3.10 -y
> conda activate llava
> pip install "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
> pip install "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git"
> pip install "transformers==4.40.0" decord   # 반드시 마지막에 구버전으로 고정 (안 되면 4.45.2 시도)
> ```
>
> 이후 `run_llava_onevision.sh`/`run_llava_video.sh`는 `llava` 환경에서, 나머지(Qwen/Gemma 4)는 기본 환경에서 실행하세요.

### HuggingFace 로그인

일부 데이터셋(Video-MME v2, LongVideoBench)과 Google 모델은 HF 계정 인증이 필요할 수 있습니다:

```bash
pip install -U "huggingface_hub[cli]"
hf auth login   # 또는 huggingface-cli login
```

게이트된(gated) 리소스는 HF 웹페이지에서 먼저 약관에 동의해야 다운로드가 됩니다.

---

## 2. 데이터셋 다운로드

**기본적으로 아무것도 미리 받을 필요가 없습니다.** lmms-eval이 첫 실행 시 HF Hub에서 자동으로 다운로드하고 압축을 풀어줍니다. 캐시 위치는 `HF_HOME`(기본 `~/.cache/huggingface`)입니다.

디스크가 작은 파티션이라면 실행 전에 캐시 위치를 옮겨두세요:

```bash
export HF_HOME=/data/hf_cache   # 원하는 큰 디스크 경로
```

미리 받아두고 싶다면 (권장 — 평가 도중 네트워크 끊김 방지):

```bash
hf download lmms-eval/Video-MME          --repo-type dataset
hf download MME-Benchmarks/Video-MME-v2  --repo-type dataset
hf download OpenGVLab/MVBench            --repo-type dataset
hf download longvideobench/LongVideoBench --repo-type dataset
hf download lmms-eval/LVBench            --repo-type dataset
hf download sy1998/MLVU_dev              --repo-type dataset
```

> **⚠️ 디스크 용량 주의**: 비디오 벤치마크는 매우 큽니다. Video-MME(~100GB), LongVideoBench/LVBench(장시간 비디오, 수백 GB) 등 **전체를 다 받으면 1TB 이상**을 잡아야 안전합니다. 벤치마크 하나씩 받아서 돌리고 지우는 방식도 가능합니다.

---

## 3. 모델 weight 다운로드

이것도 **자동**입니다. 첫 실행 시 HF Hub에서 자동으로 받아 `HF_HOME`에 캐시됩니다.

미리 받아두려면:

```bash
bash scripts/download_weights.sh          # 7B급 전 모델 + Gemma 4 E2B/E4B 일괄 다운로드
# 또는 개별로:
hf download lmms-lab/llava-onevision-qwen2-7b-ov
hf download lmms-lab/LLaVA-Video-7B-Qwen2
hf download Qwen/Qwen2-VL-7B-Instruct
hf download Qwen/Qwen2.5-VL-7B-Instruct
hf download google/gemma-4-E2B-it
hf download google/gemma-4-E4B-it
```

모델 weight 용량: 7B급 모델은 각각 약 15~17GB, Gemma 4 E2B/E4B는 각각 약 6GB/16GB(bf16 safetensors) 수준입니다. 전부 받으면 약 80~90GB입니다.

---

## 4. 평가 실행 방법

모든 스크립트는 `scripts/`에 있고, 공통 인터페이스는 다음과 같습니다:

```bash
bash scripts/run_<model>.sh [TASKS] [CHECKPOINT]
```

- `TASKS`: 쉼표로 구분한 task 이름 (생략 시 위 표의 8개 전부)
- `CHECKPOINT`: HF 체크포인트 (생략 시 7B 기본값)

### 예시

```bash
# LLaVA-OneVision 7B로 Video-MME (자막 없음/있음) 평가
bash scripts/run_llava_onevision.sh videomme,videomme_w_subtitle

# LLaVA-Video 7B로 MLVU 평가
bash scripts/run_llava_video.sh mlvu_dev

# Qwen2-VL 7B로 MVBench 평가
bash scripts/run_qwen2_vl.sh mvbench

# Qwen2.5-VL 7B로 LongVideoBench + LVBench 평가
bash scripts/run_qwen2_5_vl.sh longvideobench_val_v,lvbench

# Gemma 4 E2B로 Video-MME v2 평가
bash scripts/run_gemma4.sh videomme_v2 google/gemma-4-E2B-it

# Gemma 4 E4B로 전체 벤치마크 평가
bash scripts/run_gemma4.sh "" google/gemma-4-E4B-it

# 전 모델 × 전 벤치마크 (오래 걸립니다!)
bash scripts/run_all.sh
```

### 스크립트 없이 직접 실행하려면

스크립트는 결국 아래 형태의 명령을 실행합니다:

```bash
accelerate launch --num_processes=1 -m lmms_eval \
    --model qwen2_5_vl \
    --model_args pretrained=Qwen/Qwen2.5-VL-7B-Instruct,max_pixels=1605632,max_num_frames=32 \
    --tasks videomme \
    --batch_size 1 \
    --log_samples \
    --output_path logs/qwen2_5_vl
```

GPU가 여러 장이면 `--num_processes=N`으로 데이터 병렬 평가가 됩니다.

### 결과 확인

- 콘솔에 최종 점수 테이블이 출력됩니다.
- `logs/<model>/` 아래에 JSON 결과와 (`--log_samples` 덕분에) 샘플별 응답이 저장되어, 어떤 문제를 틀렸는지 분석할 수 있습니다.

---

## 5. Gemma 4 플러그인

lmms-eval에는 아직 Gemma 4 전용 래퍼가 없어서, 이 레포에 **플러그인 패키지**(`gemma4_plugin/`)를 만들어 두었습니다. lmms-eval의 gemma3 래퍼를 기반으로, Gemma 4의 `AutoModelForMultimodalLM` 로딩 방식에 맞게 수정한 것입니다. lmms-eval의 entry-point 메커니즘(`lmms_eval.models` 그룹)으로 등록되므로, `pip install -e gemma4_plugin` 후에는 `--model gemma4`로 바로 사용할 수 있습니다.

```bash
pip install -e gemma4_plugin   # setup.sh에 포함되어 있음
bash scripts/run_gemma4.sh videomme google/gemma-4-E4B-it
```

플러그인 등록이 안 되는 (구버전 lmms-eval 등) 경우의 수동 설치 방법:

```bash
# gemma4.py를 lmms-eval 패키지 안에 직접 복사
LMMS=$(python -c "import lmms_eval, os; print(os.path.dirname(lmms_eval.__file__))")
cp gemma4_plugin/gemma4_lmms/gemma4.py $LMMS/models/simple/gemma4.py
# 이후 $LMMS/models/__init__.py 의 simple 모델 레지스트리 dict에 "gemma4": "Gemma4" 한 줄 추가
```

또 다른 대안으로, vLLM이 Gemma 4를 지원하는 버전이라면 lmms-eval의 범용 vllm 래퍼도 사용할 수 있습니다:

```bash
python -m lmms_eval --model vllm --model_args model=google/gemma-4-E4B-it --tasks videomme ...
```

> 참고: Gemma 4 E2B/E4B는 비디오를 프레임 시퀀스로 처리하며(초당 1프레임 기준 약 60초 분량 권장), 온디바이스 지향 모델이라 장시간 비디오 벤치마크(LVBench 등)에서는 프레임 수 제한의 영향이 큽니다. 결과 해석 시 참고하세요.

---

## 6. 40GB GPU (A100 40G 등)로 가능한가?

**결론: 이 레포의 기본 설정(7B급 + E2B/E4B)은 전부 40GB 한 장으로 가능합니다.** 각 모델의 bf16 weight가 6~17GB라 여유가 있고, 관건은 weight가 아니라 **비디오 프레임이 만드는 visual token 수**입니다.

| 모델 | Weight (bf16) | 40GB 가능? | 비고 |
|---|---|---|---|
| LLaVA-OneVision 7B | ~16GB | ✅ | `max_frames_num=32` 기본. 여유 있음 |
| LLaVA-Video 7B | ~16GB | ✅ | `max_frames_num=64`도 OK (spatial pooling 덕분) |
| Qwen2-VL 7B | ~16GB | ✅ | `max_pixels`를 낮춘 상태(602112)로 실행. 기본값 그대로 장시간 비디오를 넣으면 OOM 가능 |
| Qwen2.5-VL 7B | ~17GB | ✅ | 동일. OOM 시 `max_pixels`/`max_num_frames` ↓ |
| Gemma 4 E2B | ~6GB | ✅ | 매우 여유 |
| Gemma 4 E4B | ~16GB | ✅ | 여유 |

- **OOM이 나면**: 스크립트의 `max_num_frames`(32→16), Qwen 계열은 `max_pixels`(예: 602112→301056)를 낮추세요. 점수가 약간 떨어질 수 있지만 실행은 됩니다.
- **불가능한 것**: 72B/32B급 체크포인트(예: `Qwen2-VL-72B`, `llava-onevision-qwen2-72b-ov`)는 40GB 한 장으로는 4bit 양자화 없이는 불가능합니다. 이 레포 기본 스크립트에서는 다루지 않습니다.
- 장시간 비디오 벤치마크(LVBench, MLVU, LongVideoBench)는 메모리보다 **시간**이 병목입니다. 벤치마크 하나에 수 시간~하루 이상 걸릴 수 있으니 `tmux`/`nohup` 사용을 권장합니다.

---

## 7. 트러블슈팅

| 증상 | 해결 |
|---|---|
| `decord` 관련 에러 | `pip install decord` (Mac/ARM은 `eva-decord`) |
| LLaVA 모델 로딩 실패 (`llava` import 에러) | `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git` 재설치 |
| `cannot import name 'apply_chunking_to_forward'` | transformers가 너무 최신 — 위 1번 섹션대로 LLaVA 전용 환경에서 `transformers==4.40.0`으로 고정 |
| 401/403 다운로드 에러 | `hf auth login` + 해당 HF 페이지에서 약관 동의 |
| CUDA OOM | 위 6번 섹션의 프레임/해상도 축소 참고 |
| flash-attn 빌드 실패 | 필수 아님. `attn_implementation`을 지정하지 않으면 sdpa로 동작 |
| 디스크 부족 | `export HF_HOME=<큰 디스크>` 후 재실행, 벤치마크별로 받고 지우기 |

## 라이선스 / 출처

- 평가 하니스: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (Apache 2.0 기반)
- 각 모델/데이터셋의 라이선스는 해당 HF 페이지를 따릅니다. Gemma 4는 Apache 2.0으로 공개되었습니다.
