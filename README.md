# video-llm-eval

오픈소스 Video LLM들을 주요 비디오 벤치마크에서 한 번에 검증(evaluate)하기 위한 레포입니다.
평가 하니스로는 [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval)을 사용합니다 — 아래의 모든 모델과 벤치마크를 공식 지원하므로, 이 레포는 **환경 세팅 + 실행 스크립트 + Gemma 4 플러그인 + 상세 가이드**를 제공합니다.

> 이 README의 세팅 순서와 트러블슈팅은 실제 GPU 서버(A100 40G)에서 세팅하며 겪은 에러들을 기반으로 작성되었습니다. **순서대로 따라하는 것을 권장합니다.**

## 지원 모델

| 모델 | HuggingFace 체크포인트 | 파라미터 (LM + 전체) | Vision Encoder | lmms-eval 모델 이름 | 사용 환경 |
|---|---|---|---|---|---|
| LLaVA-OneVision | `lmms-lab/llava-onevision-qwen2-7b-ov` (0.5B/7B/72B) | Qwen2-7B LM, 전체 ~8B | SigLIP-SO400M (~0.4B) | `llava_onevision` | `llava` |
| LLaVA-Video | `lmms-lab/LLaVA-Video-7B-Qwen2` (7B/72B) | Qwen2-7B LM, 전체 ~8B | SigLIP-SO400M (~0.4B) | `llava_vid` | `llava` |
| Qwen2-VL | `Qwen/Qwen2-VL-7B-Instruct` (2B/7B/72B) | Qwen2-7B LM, 전체 ~8.3B | 자체 ViT (~0.67B), naive dynamic resolution | `qwen2_vl` | `videollm` |
| Qwen2.5-VL | `Qwen/Qwen2.5-VL-7B-Instruct` (3B/7B/32B/72B) | Qwen2.5-7B LM, 전체 ~8.3B | 재설계 ViT (~0.67B), window attention | `qwen2_5_vl` | `videollm` |
| Gemma 4 E2B | `google/gemma-4-E2B-it` | 유효 2.3B / 전체 5.1B (PLE) | 자체 경량 인코더 (~0.15B), 토큰 budget 70–1120/이미지 | `gemma4` | `videollm` |
| Gemma 4 E4B | `google/gemma-4-E4B-it` | 유효 4.5B / 전체 8B (PLE) | 자체 경량 인코더 (~0.15B), 토큰 budget 70–1120/이미지 | `gemma4` | `videollm` |
| Qwen3-VL | `Qwen/Qwen3-VL-8B-Instruct` (2B/4B/8B/32B + MoE) | Qwen3-8B LM, 전체 ~9B | 자체 ViT | `qwen3_vl` | `videollm` |
| InternVL2 | `OpenGVLab/InternVL2-8B` | InternLM2.5-7B LM, 전체 ~8B | InternViT-300M | `internvl2` | `videollm` |
| InternVL2.5 | `OpenGVLab/InternVL2_5-8B` | InternLM2.5-7B LM, 전체 ~8B | InternViT-300M | `internvl2` | `videollm` |
| InternVL3 | `OpenGVLab/InternVL3-8B` | Qwen2.5-7B LM, 전체 ~8B | InternViT-300M | `internvl3` | `videollm` |
| InternVL3.5 | `OpenGVLab/InternVL3_5-8B` | Qwen3-8B LM, 전체 ~9B | InternViT-300M | `internvl3_5` | `videollm` |

> Gemma 4의 "유효(Effective) 파라미터"는 Per-Layer Embeddings(PLE) 기법으로 줄인 **메모리 기준** 수치입니다. 실제 연산량은 전체 파라미터 쪽에 가깝습니다 (아래 [속도 비교](#mvbench-실측-소요-시간-a100-40g-1장-기준) 참고).

**conda 환경이 2개 필요합니다.** LLaVA-NeXT는 구버전 transformers(4.40.0)가 필요하고, Gemma 4는 최신 transformers가 필요해서 한 환경에 공존할 수 없습니다. 자세한 건 [1. 환경 세팅](#1-환경-세팅) 참고.

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

---

## 1. 환경 세팅

### 1-0. HF_HOME 먼저 정하기 (중요!)

데이터셋/모델이 전부 `$HF_HOME` 아래에 저장되고, **HF 로그인 토큰도 `$HF_HOME/token`에 저장됩니다.**
이걸 처음에 고정하지 않으면 "받아둔 데이터셋을 못 찾음", "로그인했는데 `LocalTokenNotFoundError`" 같은 문제가 생깁니다 (tmux 창마다 셸이 새로 뜨기 때문).

```bash
df -h                    # Avail 기준 여유가 큰 디스크 확인 (전체 벤치마크는 1TB+ 권장)
export HF_HOME=<큰 디스크의 쓰기 가능한 경로>     # 예: /data/hf_cache 또는 $HOME/hf_cache
echo "export HF_HOME=$HF_HOME" >> ~/.bashrc      # 영구 고정 — 반드시 할 것
```

- `/data` 같은 경로에서 `PermissionError: [Errno 13]`이 나면: `sudo mkdir -p /data/hf_cache && sudo chown -R $USER /data/hf_cache`, sudo가 없으면 홈이나 scratch 디스크 사용.
- 현재 캐시가 어디에 뭘 갖고 있는지는 `hf cache scan`으로 언제든 확인 가능.

### 1-1. conda 환경 ① `videollm` (Qwen2-VL / Qwen2.5-VL / Gemma 4)

```bash
git clone https://github.com/dnwjddl/video-llm-eval.git
cd video-llm-eval

conda create -n videollm python=3.10 -y
conda activate videollm
bash setup.sh
```

- `CondaToSNonInteractiveError` (Terms of Service 에러)가 나면:
  ```bash
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main
  conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r
  ```
  또는 채널 자체를 우회: `conda create -n videollm python=3.10 -y -c conda-forge --override-channels`

### 1-2. conda 환경 ② `llava` (LLaVA-OneVision / LLaVA-Video)

```bash
conda create -n llava python=3.10 -y
conda activate llava
pip install "git+https://github.com/LLaVA-VL/LLaVA-NeXT.git"
pip install "git+https://github.com/EvolvingLMMs-Lab/lmms-eval.git"
pip install "transformers==4.40.0" decord     # 반드시 마지막에! (설치 순서 중요)
```

- URL을 손으로 치지 말고 **그대로 복사**하세요. `LLaVA-VL/`(org 부분)이 빠지면 clone 실패합니다.
- `transformers==4.40.0` 설치 시 pip이 의존성 충돌 경고(sentence-transformers 등)를 띄우는데 **무시해도 됩니다.** 판단 기준은 경고가 아니라 아래 1-4의 import 확인입니다.
- 4.40.0에서 lmms-eval import가 실패하면 `4.45.2`로 올려서 재시도.

### 1-3. PyTorch ↔ CUDA 드라이버 맞추기

`RuntimeError: The NVIDIA driver on your system is too old`가 나면, pip이 설치한 최신 torch가 서버 드라이버보다 새로운 CUDA로 빌드된 것입니다. **드라이버 버전에 맞는 torch를 설치**하세요 (두 환경 모두 확인!):

```bash
nvidia-smi                # 우측 상단 "CUDA Version"이 드라이버가 지원하는 최대 버전
# 예: CUDA Version 12.4 이면 →
pip uninstall -y torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
# (12.1이면 cu121, 12.6이면 cu126 …)
```

### 1-4. 설치 확인 체크리스트

```bash
# 두 환경 공통
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"   # True 필수
python -c "import lmms_eval; print('lmms_eval OK')"

# llava 환경
python -c "import transformers; print(transformers.__version__)"                # 4.40.0
python -c "from llava.model.language_model.llava_llama import LlavaLlamaForCausalLM; print('llava OK')"

# videollm 환경
python -c "import transformers; print(transformers.__version__)"                # 최신 (4.40대면 pip install -U transformers)
python -c "from gemma4_lmms.gemma4 import Gemma4; print('gemma4 plugin OK')"
```

### 1-5. HuggingFace 로그인

`LocalTokenNotFoundError`, `HfHubHTTPError` 방지. **HF_HOME이 설정된 셸에서** 로그인하세요:

```bash
echo $HF_HOME             # 1-0에서 정한 경로가 나와야 함 (안 나오면 source ~/.bashrc)
hf auth login             # 구버전 hub이면: huggingface-cli login
# 토큰은 https://huggingface.co/settings/tokens 에서 생성 (Read 권한이면 충분)
hf auth whoami            # 사용자명이 나오면 성공
```

게이트된(gated) 데이터셋(Video-MME v2, LongVideoBench 등)은 HF 웹페이지에서 약관 동의도 필요합니다.

---

## 2. 데이터셋 다운로드

lmms-eval이 첫 실행 시 자동으로 받지만, **미리 받아두는 것을 권장**합니다 (평가 도중 네트워크 실패 방지):

```bash
hf download lmms-eval/Video-MME           --repo-type dataset
hf download MME-Benchmarks/Video-MME-v2   --repo-type dataset
hf download OpenGVLab/MVBench             --repo-type dataset
hf download longvideobench/LongVideoBench --repo-type dataset
hf download lmms-eval/LVBench             --repo-type dataset
hf download sy1998/MLVU_dev               --repo-type dataset
```

> **⚠️ 디스크 용량 주의**: 전체를 다 받으면 1TB 이상이 필요합니다. 여유가 300~400GB 수준이면 **순환 방식**으로 진행하세요: 벤치마크 하나 받고 → 전 모델 평가 → `hf cache delete`로 지우고 → 다음 것. 첫 평가 때 zip 압축 해제로 공간이 추가로 필요하다는 것도 계산에 넣으세요. 진행 중 수시로 `df -h`로 확인.

- 다운로드가 끊기면 같은 명령을 다시 실행 — 이어받기 됩니다.
- 레포 이름 오타 주의: `lmms-eval` (m 두 개), `lms-eval` 아님.
- **미리 받아두면 평가는 `HF_HUB_OFFLINE=1`로 실행하세요** (예: `HF_HUB_OFFLINE=1 bash scripts/run_llava_onevision.sh mvbench`). 평가 중 HF API를 안 건드려서 429 rate limit을 원천 차단합니다.

## 3. 모델 weight 다운로드

첫 실행 시 자동 다운로드되지만, 미리 받으려면:

```bash
bash scripts/download_weights.sh     # 전 모델 일괄 (~90GB)
```

---

## 4. 모델별 사용법

공통 사항:

- 스크립트 인터페이스: `bash scripts/run_<model>.sh [TASKS] [CHECKPOINT]` — TASKS 생략(또는 `""`) 시 8개 벤치마크 전부.
- 오래 걸리므로 **tmux 안에서** 실행하세요. 결과는 `logs/<model>/`에 저장 (`--log_samples`라 샘플별 응답까지 남음).
- 멀티 GPU면 `NUM_GPUS=4 bash scripts/...`로 데이터 병렬, 특정 GPU 지정은 `CUDA_VISIBLE_DEVICES=1 bash scripts/...`.
- **본 평가 전에 항상 스모크 테스트** (`mvbench` 하나만) 먼저 — 3일 돌린 뒤 발견하는 것보다 10분 만에 발견하는 게 낫습니다.

### 4-1. LLaVA-OneVision (`llava` 환경)

```bash
conda activate llava
bash scripts/run_llava_onevision.sh mvbench                    # 스모크 테스트
bash scripts/run_llava_onevision.sh videomme,videomme_w_subtitle
```

- 기본 설정: 7B 체크포인트, 32프레임, `conv_template=qwen_1_5`.
- 프레임 수를 바꾸려면 스크립트의 `max_frames_num`을 수정 (OOM 시 16으로).

### 4-2. LLaVA-Video (`llava` 환경)

```bash
conda activate llava
bash scripts/run_llava_video.sh mlvu_dev
```

- 기본 설정: 64프레임 + average spatial pooling (공식 평가 세팅).
- LLaVA-OneVision과 같은 환경/의존성을 공유하므로 OneVision이 돌면 이것도 돕니다.

### 4-3. Qwen2-VL (`videollm` 환경)

```bash
conda activate videollm
bash scripts/run_qwen2_vl.sh mvbench
```

- 기본 설정: `max_pixels=602112` (40GB OOM 방지용으로 낮춰둠), 32프레임.
- OOM 시: `max_pixels=301056`, `max_num_frames=16`으로 낮추기. 해상도/프레임을 낮추면 점수가 약간 떨어질 수 있으나 실행은 됩니다.

### 4-4. Qwen2.5-VL (`videollm` 환경)

```bash
conda activate videollm
bash scripts/run_qwen2_5_vl.sh longvideobench_val_v,lvbench
```

- 기본 설정: `max_pixels=1605632`, 32프레임. OOM 시 Qwen2-VL과 동일하게 낮추세요.
- 장시간 비디오 벤치마크(LVBench 등)에서 특히 메모리를 많이 씁니다.

### 4-5. Gemma 4 E2B / E4B (`videollm` 환경)

```bash
conda activate videollm
bash scripts/run_gemma4.sh mvbench google/gemma-4-E2B-it       # E2B 스모크 테스트
bash scripts/run_gemma4.sh "" google/gemma-4-E2B-it            # E2B 전체 벤치마크
bash scripts/run_gemma4.sh "" google/gemma-4-E4B-it            # E4B 전체 벤치마크
```

- 이 레포의 **플러그인**(`gemma4_plugin/`)으로 동작합니다 — lmms-eval에 아직 Gemma 4 전용 래퍼가 없어서, gemma3 래퍼를 `AutoModelForMultimodalLM` 로딩에 맞게 수정해 entry-point로 등록한 것입니다. `setup.sh`가 자동 설치하며, 결과는 `logs/gemma-4-E2B-it/`처럼 체크포인트명으로 저장됩니다.
- 플러그인 인식이 안 되면: `pip install -e gemma4_plugin` 재실행 후 1-4의 plugin OK 확인.
- Gemma 4는 최신 transformers가 필요합니다. `llava` 환경에서 돌리면 안 됩니다.
- E2B/E4B는 온디바이스 지향 모델이라 장시간 비디오 벤치마크에서 프레임 수 제한의 영향이 큽니다. 결과 해석 시 참고.

### 4-6. Qwen3-VL / InternVL 계열 (`videollm` 환경)

```bash
conda activate videollm

# Qwen3-VL 8B
hf download Qwen/Qwen3-VL-8B-Instruct
HF_HUB_OFFLINE=1 bash scripts/run_qwen3_vl.sh mvbench

# InternVL — 체크포인트 이름으로 세대(래퍼) 자동 선택
hf download OpenGVLab/InternVL3-8B
HF_HUB_OFFLINE=1 bash scripts/run_internvl.sh mvbench OpenGVLab/InternVL3-8B
# 다른 세대: OpenGVLab/InternVL2-8B, OpenGVLab/InternVL2_5-8B, OpenGVLab/InternVL3_5-8B
```

- InternVL은 `trust_remote_code` 기반 커스텀 코드라 transformers 버전을 탑니다. `videollm` 환경(최신 transformers)에서 구세대(InternVL2/2.5)가 import 에러를 내면, `llava` 환경(transformers 4.40.0)에서 시도해보세요.
- InternVL3/3.5는 flash-attn이 기본이지만 스크립트에서 `use_flash_attn=False`로 꺼뒀습니다 (미설치 환경 대비). flash-attn을 설치했다면 스크립트에서 True로 바꾸면 더 빠릅니다.

### 전 모델 일괄 실행

```bash
bash scripts/run_all.sh    # 단, llava/videollm 환경 분리 때문에 LLaVA 계열은 llava 환경에서 별도 실행 필요
```

---

## 5. 40GB GPU (A100 40G 등)로 가능한가?

**결론: 이 레포의 기본 설정(7B급 + E2B/E4B)은 전부 40GB 한 장으로 가능합니다.** 관건은 weight(6~17GB)가 아니라 **비디오 프레임이 만드는 visual token 수**입니다.

| 모델 | Weight (bf16) | 40GB 가능? | 비고 |
|---|---|---|---|
| LLaVA-OneVision 7B | ~16GB | ✅ | 32프레임 기본. 여유 있음 |
| LLaVA-Video 7B | ~16GB | ✅ | 64프레임도 OK (spatial pooling 덕분) |
| Qwen2-VL 7B | ~16GB | ✅ | `max_pixels` 낮춘 상태로 실행 |
| Qwen2.5-VL 7B | ~17GB | ✅ | OOM 시 `max_pixels`/프레임 ↓ |
| Gemma 4 E2B | ~6GB | ✅ | 매우 여유 |
| Gemma 4 E4B | ~16GB | ✅ | 여유 |

- **불가능한 것**: 72B/32B급 체크포인트는 40GB 한 장으로는 4bit 양자화 없이 불가.
- 장시간 비디오 벤치마크(LVBench, MLVU, LongVideoBench)는 메모리보다 **시간**이 병목 — 벤치마크 하나에 수 시간~하루 이상. 같은 GPU에 두 모델을 동시에 올리지 마세요 (OOM).

### MVBench 실측 소요 시간 (A100 40G 1장 기준)

MVBench 4,000문항 기준 실측 wall-clock입니다 (1회 측정치라 오차 있음):

| 모델 | 소요 시간 | 문항당 | 비고 |
|---|---|---|---|
| LLaVA-Video 7B | ~50분 | ~0.75초 | 64프레임인데도 spatial pooling 덕분에 빠름 |
| Qwen2-VL 7B | ~50분 | ~0.75초 | `max_pixels` 낮춘 설정 기준 |
| LLaVA-OneVision 7B | ~2시간 45분 | ~2.5초 | **첫 실행이라 MVBench 데이터셋 전처리(압축 해제 + 캐시 빌드) 시간이 포함됨** — 순수 추론은 이보다 빠를 것 |
| Gemma 4 E2B | ~11시간 15분 | ~10초 | 아래 참고 |

#### 왜 파라미터가 제일 적은 E2B가 제일 느린가?

파라미터 수와 추론 속도는 별개입니다. E2B가 느린 이유:

1. **"유효 2.3B"는 메모리 기준이지 연산량 기준이 아님.** PLE(Per-Layer Embeddings)는 "2B급 메모리 사용량"을 만드는 기법이고, 실제 파라미터는 5.1B, 연산량(FLOPs)도 그에 가깝습니다. 온디바이스에서 "메모리에 들어가느냐"를 위한 설계라 GPU 서버에서의 속도 이점은 이름만큼 크지 않습니다.
2. **추론 경로의 최적화 수준 차이 (가장 큰 요인).** LLaVA 계열은 LLaVA-NeXT 레포의 손질된 경로(효율적 프레임 토큰 풀링, flash-attn/SDPA 자동 선택, decord 디코딩)를 타고, Qwen 계열도 transformers에 오랫동안 최적화가 쌓여 있습니다. 반면 Gemma 4는 이 레포의 플러그인이 transformers **범용 경로**로 돌리며, 출시된 지 얼마 안 된 아키텍처라 커널 최적화도 아직 덜 되어 있습니다 (attention 기본값이 비효율적인 eager로 잡힐 수 있음).
3. **프레임당 비주얼 토큰 수 차이.** LLaVA 계열은 pooling으로 프레임 토큰을 강하게 압축하지만, Gemma 4 프로세서의 토큰 budget 설정에 따라 같은 32프레임이라도 시퀀스가 훨씬 길어질 수 있습니다.

속도 개선 실험: `scripts/run_gemma4.sh`의 model_args에 `attn_implementation=sdpa`를 추가해보세요. 벤치마크 점수에는 영향 없이 속도만 달라집니다.

---

## 6. 트러블슈팅 (전부 실제로 겪은 에러들)

| 증상 | 원인 | 해결 |
|---|---|---|
| `PermissionError: [Errno 13] '/data'` | HF_HOME 경로에 쓰기 권한 없음 | `sudo chown -R $USER <경로>` 또는 쓰기 가능한 경로로 HF_HOME 변경 ([1-0](#1-0-hf_home-먼저-정하기-중요)) |
| `CondaToSNonInteractiveError` | Anaconda 기본 채널 약관 미동의 | `conda tos accept ...` 또는 conda-forge 사용 ([1-1](#1-1-conda-환경--videollm-qwen2-vl--qwen25-vl--gemma-4)) |
| `cannot import name 'apply_chunking_to_forward'` | transformers가 LLaVA-NeXT 기준으로 너무 최신 | `llava` 환경에서 `transformers==4.40.0` 고정 ([1-2](#1-2-conda-환경--llava-llava-onevision--llava-video)) |
| pip "dependency resolver" 빨간 경고 | 미사용 패키지(sentence-transformers 등)의 버전 불평 | **무시**. 기준은 1-4 import 체크리스트 통과 여부 |
| `RuntimeError: NVIDIA driver ... too old` | torch가 드라이버보다 새 CUDA로 빌드됨 | 드라이버에 맞는 빌드 재설치 (예: `--index-url .../whl/cu124`) ([1-3](#1-3-pytorch--cuda-드라이버-맞추기)) |
| `LocalTokenNotFoundError` | HF 미로그인, 또는 로그인한 셸과 HF_HOME 불일치 | HF_HOME 고정 후 그 셸에서 `hf auth login` ([1-5](#1-5-huggingface-로그인)) |
| `HfHubHTTPError` / `RetryError` | 토큰 401, gated 약관 미동의(403), 요청 과다(429) 등 | `hf auth whoami` 확인 → 안 되면 `--verbosity=DEBUG`로 HTTP 코드 확인 |
| `429 Too Many Requests` (5분당 1000 요청 한도) | 평가 중 스트리밍 다운로드로 API 요청 폭증 | 5~10분 대기 → `hf download <데이터셋> --repo-type dataset`으로 **미리 통째로 받고** → `HF_HUB_OFFLINE=1 bash scripts/...`로 평가 |
| 받아둔 데이터셋이 안 보임 / 재다운로드 시작 | 셸마다 HF_HOME이 다름 | `hf cache scan`으로 실제 캐시 위치 확인 후 `.bashrc`에 HF_HOME 고정 |
| `EnvironmentNameNotFound` (source ~/.bashrc 시) | `.bashrc`에 존재하지 않는 conda 환경 activate 줄 | 해당 줄 삭제: `sed -i '/conda activate <이름>/d' ~/.bashrc` |
| `git clone ... 실패` (pip install git+...) | URL에서 org 부분(`LLaVA-VL/`) 누락 | 명령을 그대로 복사해서 실행 |
| CUDA OOM | 비디오 프레임 토큰 과다 | `max_num_frames`/`max_pixels` 낮추기 ([5](#5-40gb-gpu-a100-40g-등로-가능한가)) |
| `decord` 에러 | 미설치 | `pip install decord` |
| 디스크 부족 | 데이터셋 용량 | 순환 방식 + `hf cache delete` ([2](#2-데이터셋-다운로드)) |

### tmux 최소 사용법

```bash
tmux new -s eval          # 세션 생성
# Ctrl+b "  → 위아래 분할 | Ctrl+b %  → 좌우 분할 | Ctrl+b 방향키 → 창 이동
# Ctrl+b z  → 현재 창 확대 토글 | Ctrl+b d → 세션에서 나가기(작업은 계속 돌아감)
tmux attach -t eval       # 재접속
tmux set -g mouse on      # 마우스로 창 전환/크기조절/스크롤 (복사는 Shift+드래그)
```

**tmux 새 창을 열면**: `HF_HOME`은 `.bashrc` 덕에 자동으로 잡히지만, **conda 환경은 매번 직접 activate** 해야 합니다.

## 결과 (진행 중)

A100 40G 1장, 이 레포의 기본 스크립트 설정으로 측정한 결과입니다. 괄호는 공식 보고치.

| 모델 | MVBench | Video-MME (w/o sub) | Video-MME (w/ sub) | Video-MME v2 | v2 (w/ sub) | LongVideoBench | LVBench | MLVU |
|---|---|---|---|---|---|---|---|---|
| LLaVA-OneVision 7B | **58.35** (56.7) | | | | | | | |
| LLaVA-Video 7B | **60.38** (58.6) | | | | | | | |
| Qwen2-VL 7B | **66.30** (67.0) | | | | | | | |
| Qwen2.5-VL 7B | | | | | | | | |
| Gemma 4 E2B | | | | | | | | |
| Gemma 4 E4B | | | | | | | | |

두 모델 모두 공식 보고치 대비 +1.7점 내외로 일관되게 측정됨 — 프레임 수/프롬프트 세부 설정 차이에 의한 정상 범위 편차.

### 점수 추출

```bash
bash scripts/score.sh              # 모든 모델/벤치마크 결과 요약
bash scripts/score.sh qwen2_vl     # 경로에 qwen2_vl이 들어간 결과만
```

logs/ 아래 모든 결과 파일에서 벤치마크별 점수를 뽑아주고, MVBench는 서브태스크 20개 평균(= 최종 점수)을 `>>` 줄로 자동 계산합니다.

## 라이선스 / 출처

- 평가 하니스: [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) (Apache 2.0 기반)
- 각 모델/데이터셋의 라이선스는 해당 HF 페이지를 따릅니다. Gemma 4는 Apache 2.0으로 공개되었습니다.
