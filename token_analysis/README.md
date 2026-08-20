# Token Analysis — 비주얼 토큰 압축 ablation

**질문**: 비디오 LLM의 비주얼 토큰을 얼마나, 어떤 방식으로 줄여도 되는가?
토큰 유지 비율을 스윕하면서 (1) 방법별 정확도 곡선, (2) temporal vs spatial 태스크의 열화 차이를 측정한다.

## 실험 설계

- **실험대 모델**: LLaVA-OneVision-7B. 비주얼 토큰이 1D 시퀀스로 단순 삽입되는 구조라
  토큰 수를 임의로 바꿔도 위치 인코딩이 깨지지 않는다 (Qwen 계열은 M-RoPE가 토큰
  격자에 묶여 있어 압축 방법 비교의 교란 변수가 됨).
- **개입 지점**: lmms-eval이 쓰는 llava의 `get_2dPool`(프레임당 729토큰 → 풀링)을
  통째로 대체 — **모든 방법이 같은 입력(729토큰/프레임)에서 출발**하므로 공정 비교.
- **채점**: lmms-eval 파이프라인을 in-process로 그대로 호출 (MVBench).

## 구현된 방법 (`compress.py`)

| 축 | 방법 | 설명 |
|---|---|---|
| 공간 풀링 | `pool_avg`, `pool_max` | 격자 pooling (모델 기본값의 일반화) |
| 무작위 기준선 | `random` | 토큰 무작위 유지 — 모든 방법의 하한 기준 |
| PCA-선택 | `pca_select` | 프레임별 주성분 leverage score 상위 토큰 유지 |
| PCA-재구성 | `pca_recon` | **토큰 수 유지**, 상위 rank만 남겨 재구성 — "정보량" 축의 ablation (토큰 수 축과 직교) |
| 병합 | `tome` | ToMe식 bipartite soft matching (유사 토큰 크기가중 병합) |
| 군집 | `kmeans` | 프레임별 k-means 센터로 대체 |
| 시간 풀링 | `temporal_pool` | 인접 프레임 평균 (프레임당 토큰 유지) |
| 시간 선택 | `framediff` | 변화량 큰 프레임만 유지 — temporal 정보 보존 가설 검증용 |

## 실행

```bash
conda activate llava
# 단일 실험
python token_analysis/run_token_ablation.py --method pca_select --keep 0.25
# 전체 스윕 (방법 8종 × 비율 4종 + 기준선 + rank ablation) — 오래 걸림, tmux 권장
bash token_analysis/run_sweep.sh
# 그래프 (overall / temporal / spatial 3패널 곡선)
python token_analysis/plot_ablation.py
```

- 기본 태스크는 대표 8개 서브태스크 (temporal 4 + spatial 4)로 스윕 비용을 줄였다.
  결론이 나온 방법만 `--tasks mvbench`로 20개 전체 검증 권장.
- temporal/spatial 구분은 `plot_ablation.py`의 `TEMPORAL` 집합 (우리 정의) — 필요시 수정.

## 기대하는 분석 산출물

1. **방법별 열화 곡선**: keep ratio ↓에 따른 정확도 — 어떤 방법이 같은 예산에서 우월한가
2. **temporal 취약성**: spatial 태스크는 버티는데 temporal 태스크가 먼저 무너지는 지점
   → "공간 중심 압축(pool/PCA/ToMe)은 temporal 정보를 놓친다"의 정량 증거
3. **rank vs count**: `pca_recon`(정보량 축소) vs `pca_select`(토큰 수 축소) 비교 —
   성능을 결정하는 게 토큰 "개수"인지 "정보량"인지 분리
4. **random 대비 이득**: 각 방법이 무작위 기준선보다 실제로 얼마나 나은가

## 관련 연구 (비교/인용 후보)

| 방법 | 아이디어 | 비고 |
|---|---|---|
| [ToMe](https://arxiv.org/abs/2210.09461) | bipartite 토큰 병합 | 우리 `tome`의 원형 |
| FastV | LLM 얕은 층의 attention으로 프루닝 | training-free 프루닝 대표 |
| VisionZip | 인코더 attention 기반 선택 | |
| [LongVU](https://arxiv.org/abs/2410.17434) | 시공간 적응 압축 (장시간 비디오) | |
| DyCoke / DynTok | 동적 토큰 압축 (비디오) | "DynaTok"으로 언급된 것이 이 중 하나일 가능성 |
| TempMe, PruneVID, FastVID | 프레임 간 유사 토큰 병합/구간 선택 | temporal 중복 활용 계열 |
| HoliTom | 전역(holistic) 중복까지 고려한 병합 | |
| [LiteFrame](https://arxiv.org/abs/2605.17260) | 프레임당 16토큰 고정 WAP(4,2,2), 인코더 쪽 접근 | 2026, 공개 repo는 placeholder 상태 |
| InfoMerge, OTT-Vid | 정보량/최적수송 기반 병합 | 2026 |
| [서베이 (TMLR 2026)](https://arxiv.org/abs/2507.20198) | 토큰 압축 전반 정리 | [awesome list](https://github.com/cokeshao/Awesome-Multimodal-Token-Compression) |

우리 실험은 위 방법들의 "학습 없는 공통 골격"(pool/select/merge/cluster)을 같은 조건에서
직접 비교하는 controlled study 포지션 — 특정 방법 재현이 아니라 **설계 축의 분리**가 목적.
