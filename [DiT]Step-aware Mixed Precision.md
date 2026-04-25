### 10 Steps에서 Quant 낮추기 (Step-aware Mixed Precision)

* SVDQuant 4-bit이 FP16 대비 이미 ~2–2.5× 속도 이득을 내는 상태고,
* 여기서 3-bit이나 2-bit로 내려가도 kernel 효율상 추가 이득이 10–20% 수준입니다
    * (메모리 bandwidth는 줄지만 compute는 크게 안 줄어듦).

* Current: SVDQuant 4-bit, 10 steps, 1.44s
* Mixed 3/4-bit (민감 step만 4-bit, 나머지 3-bit):
    * ~1.25–1.30s 기대. 1.10–1.15× 추가 gain
    * 2-bit까지 내리면 kernel 효율 떨어져서 오히려 느려질 가능성 있음.
    * Inference speed 관점에서는 W4A4 이하는 실익이 적음.

* 이 방향의 진짜 가치는 speedup이 아니라 메모리 footprint입니다.
* Edge device 배포 / batch size 늘리기 용도.
* NeurIPS 논문 angle로는 "quant의 timestep 민감도 분석" 자체가 기여.

---

# Step-aware Mixed Precision — Experiment Results

**Model**: PixArt-Sigma-XL-2-1024-MS | **Steps**: 10 | **Samples**: n=100 (MJHQ-30K)  
**Quant**: Fake-quant (dequant→FP16). Memory savings = analytical only.  
**Date**: 2026-04-24

---

## Bounds

| Config | FID ↓ | CLIP ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | TPI (s) | MB | Save |
|---|---|---|---|---|---|---|---|---|
| W4A4 (baseline) | **124.1** | 34.53 | 14.64 | 0.543 | 0.480 | 4.72 | 297.3 | 0% |
| W3A3 (all) | 391.7 | 23.50 | 9.76 | 0.252 | 0.783 | 1.06 | 223.0 | 25% |

---

## Phase 2 — Sensitivity Heatmap (latent MSE per layer_type × step)

단일 (layer_type, step) pair를 W4A4→W3A3으로 전환했을 때의 latent MSE.  
**낮을수록 tolerant** (W3A3 적용 적합).

| layer_type | s0 | s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8 | s9 |
|---|---|---|---|---|---|---|---|---|---|---|
| attn1_qkv | 0.151 | **0.199** | 0.165 | 0.156 | 0.115 | 0.089 | 0.071 | 0.065 | 0.082 | 0.070 |
| attn1_out | 0.148 | 0.190 | 0.152 | 0.117 | 0.083 | 0.072 | 0.059 | 0.050 | 0.044 | 0.025 |
| attn2_qkv | 0.163 | 0.169 | 0.153 | 0.104 | 0.086 | 0.070 | 0.056 | 0.048 | 0.039 | 0.018 |
| attn2_out | 0.140 | 0.120 | 0.104 | 0.091 | 0.077 | 0.062 | 0.054 | 0.045 | 0.037 | **0.017** |
| mlp_fc1 | 0.153 | 0.205 | **0.207** | 0.139 | 0.107 | 0.083 | 0.066 | 0.057 | 0.058 | 0.041 |
| mlp_fc2 | 0.176 | **0.228** | 0.144 | 0.126 | 0.092 | 0.072 | 0.059 | 0.052 | 0.050 | 0.030 |

- **s9이 전체적으로 가장 tolerant**, s0~s2가 가장 sensitive
- **attn2_out이 가장 tolerant layer type** (row 평균 최저)

---

## Phase 3 — S1: top-K tolerant (layer_type × step) pairs → W3A3

| K | FID ↓ | CLIP ↑ | PSNR ↑ | SSIM ↑ | LPIPS ↓ | TPI (s) | MB | Save |
|---|---|---|---|---|---|---|---|---|
| 0 | **124.1** | 34.53 | 14.64 | 0.543 | 0.480 | 4.72 | 297.3 | 0% |
| 10 | 186.2 | 34.01 | 14.40 | 0.507 | 0.549 | 4.46 | 287.1 | 3.4% |
| 20 | 237.0 | 32.43 | 14.33 | 0.492 | 0.582 | 3.79 | 274.5 | 7.7% |
| 30 | 328.9 | 28.65 | 13.24 | 0.395 | 0.678 | 3.51 | 262.4 | 11.7% |
| 40 | 359.5 | 26.59 | 12.14 | 0.334 | 0.714 | 2.93 | 249.9 | 15.9% |
| 50 | 362.5 | 25.48 | 11.33 | 0.309 | 0.746 | 2.26 | 237.8 | 20.0% |
| 60 | 391.7 | 23.50 | 9.76 | 0.252 | 0.783 | 1.06 | 223.0 | 25.0% |

K=10만으로도 FID 124→186. "free lunch" 없음.

---

## Phase 4 — S2 / S3 Schedule Variants

### S2: step-uniform (K개 step 전체 → W3A3)

| K steps | FID ↓ | CLIP ↑ | TPI (s) | MB | Save |
|---|---|---|---|---|---|
| 2 | 294.0 | 31.33 | 4.74 | 282.4 | 5.0% |
| 4 | 335.4 | 28.83 | 4.04 | 267.5 | 10.0% |
| 6 | 356.5 | 26.81 | 2.78 | 252.7 | 15.0% |
| 8 | 381.6 | 24.32 | 1.75 | 237.8 | 20.0% |

S1보다 나쁨. step uniform은 비효율적.

### S3: layer-type-uniform (K개 layer type 전체 step → W3A3)

tolerance order: attn2_out < attn2_qkv < attn1_out < attn1_qkv < mlp_fc1 < mlp_fc2

| K | Types (W3A3) | FID ↓ | CLIP ↑ | PSNR ↑ | TPI (s) | MB | Save |
|---|---|---|---|---|---|---|---|
| 1 | attn2_out | **120.6** | 34.31 | 14.45 | 4.18 | 292.6 | 1.6% |
| 2 | +attn2_qkv | 144.6 | 34.54 | 13.21 | 3.78 | 278.7 | 6.2% |
| 3 | +attn1_out | 163.0 | 34.16 | 12.87 | 3.46 | 274.0 | 7.8% |
| 4 | +attn1_qkv | 215.3 | 32.62 | 12.06 | 2.29 | 255.5 | 14.1% |
| 5 | +mlp_fc1 | 296.7 | 28.48 | 11.15 | 1.99 | 236.9 | 20.3% |
| 6 | +mlp_fc2 | 391.7 | 23.50 | 9.76 | 1.07 | 223.0 | 25.0% |

**S3 k=1 (attn2_out 전체 step W3A3)** = 유일하게 품질 유지 (FID 124→120, −1.6%).

---

## Phase 5 — S4: per-block top-K (280 pairs)

28 blocks × 10 steps = 280 (block, step) pairs 측정. 가장 tolerant는 후반 block × 후반 step.

| K | FID ↓ | CLIP ↑ | PSNR ↑ | TPI (s) | MB | Save |
|---|---|---|---|---|---|---|
| 20 | 168.0 | 34.33 | 14.49 | 4.20 | ~299.8 | ~1.7% |
| 40 | 243.9 | 32.95 | 14.05 | 3.97 | ~294.5 | ~3.5% |
| 80 | 285.5 | 31.33 | 13.91 | 3.48 | 283.9 | 7.0% |
| 140 | 335.2 | 27.96 | 13.12 | 2.74 | 267.9 | 12.2% |
| 200 | 362.2 | 25.86 | 11.69 | 2.02 | 252.0 | 17.4% |
| 280 | 391.7 | 23.50 | 9.76 | 1.04 | 230.8 | 24.4% |

S3 k=1 (FID=120.6 @ 1.6%)보다 S4 K=20 (FID=168 @ 1.7%)이 나쁨. **block 세분화 이득 없음**.

---

## Phase 6 — Greedy Schedule (joint optimization)

Forward greedy: 매 step마다 combined MSE 최소화 pair 추가. K=20까지 탐색.

선택 순서: step=9 완전 소진 → step=8 → step=7 → step=6 → step=5 순으로 진행.  
(S1 top-K ranking과 거의 동일 — W3A3 pair interaction이 additive함을 확인)

### FID 평가 (K=10, K=20)

| K | Method | FID ↓ | CLIP ↑ | PSNR ↑ | TPI (s) | MB | Save |
|---|---|---|---|---|---|---|---|
| 10 | **Greedy** | **156.6** | 34.36 | 14.54 | 3.94 | 287.1 | 3.4% |
| 10 | S1 top-K | 186.2 | 34.01 | 14.40 | 4.46 | 287.1 | 3.4% |
| 20 | **Greedy** | **186.9** | 33.79 | 14.51 | 3.45 | 276.4 | 7.0% |
| 20 | S1 top-K | 237.0 | 32.43 | 14.33 | 3.79 | 274.5 | 7.7% |

Greedy가 K=10에서 −30, K=20에서 −50 FID 개선. 단 두 방법 모두 품질 붕괴 수준.

---

## Phase 7 — Bit-decoupling Ablation

W4A4→W3A3 cliff의 원인 분리: weight 3-bit vs activation 3-bit.

| Config | Weight | Act | FID ↓ | CLIP ↑ | PSNR ↑ | TPI (s) | MB |
|---|---|---|---|---|---|---|---|
| W4A4 | NVFP4 | NVFP4 | **124.1** | 34.53 | 14.64 | 4.72 | 297.3 |
| W4A3 | NVFP4 | INT3 | 358.2 | 26.15 | 11.14 | 1.46 | 297.3 |
| W3A4 | INT3 | NVFP4 | 185.8 | 32.99 | 11.37 | 4.48 | 223.0 |
| **W4FP16** | NVFP4 | FP16 | **89.4** | 34.55 | 14.87 | **0.44** | 297.3 |
| W3FP16 | INT3 | FP16 | 157.7 | 33.56 | 10.99 | 0.45 | 223.0 |
| W3A3 | INT3 | INT3 | 391.7 | 23.50 | 9.76 | 1.06 | 223.0 |

- **INT3 activation이 cliff 주범**: W4A3 (358) ≈ W3A3 (392) >> W3A4 (186)
- **W4FP16**: activation quant 제거 시 FID=89.4, TPI=0.44s (8× 속도). fake-quant act overhead가 TPI 대부분을 차지함.

---

## Phase 8 — DeepCache + Mixed Precision

DeepCache: blocks 8–19를 짝수 step에서 cache reuse (cache_start=8, end=20, interval=2).

| Schedule | DeepCache | FID ↓ | CLIP ↑ | PSNR ↑ | TPI (s) | MB | Save |
|---|---|---|---|---|---|---|---|
| W4A4 | Off | **124.1** | 34.53 | 14.64 | 4.72 | 297.3 | 0% |
| W4A4 | On | 139.4 | 34.52 | 14.20 | 3.53 | 297.3 | 0% |
| S3-k1 | Off | **120.6** | 34.31 | 14.45 | 4.18 | 292.6 | 1.6% |
| **S3-k1** | **On** | **139.9** | 34.22 | 14.00 | **3.30** | 292.6 | **1.6%** |

DeepCache로 TPI −25%, FID +15. S3-k1과 병용 시 추가 TPI −9% (총 −30% vs baseline).

---

## 종합 결론

| Method | FID ↓ | TPI (s) | MB | Save | 비고 |
|---|---|---|---|---|---|
| W4A4 baseline | 124.1 | 4.72 | 297.3 | 0% | 기준 |
| **S3-k1** | **120.6** | 4.18 | 292.6 | 1.6% | 유일한 품질 보존 config |
| **S3-k1 + DeepCache** | 139.9 | **3.30** | 292.6 | 1.6% | TPI 최적 Pareto ⭐ |
| W4A4 + DeepCache | 139.4 | 3.53 | 297.3 | 0% | DeepCache 단독 효과 |
| Greedy K=10 | 156.6 | 3.94 | 287.1 | 3.4% | S1 K=10보다 −30 FID |
| W3FP16 | 157.7 | 0.45 | 223.0 | 25% | 속도 최우선 시 |
| **W4FP16** | **89.4** | **0.44** | 297.3 | 0% | 품질+속도 최고 (act quant 제거) |
| all-W3A3 | 391.7 | 1.06 | 223.0 | 25% | 붕괴 |

**핵심 3가지:**
1. INT3 activation이 품질 붕괴의 주원인. weight INT3은 moderate 손실.
2. W3A3 mixed precision은 S3-k1 (1.6% save) 이상 적용 불가 — 이후는 급격한 FID 붕괴.
3. TPI 절감은 DeepCache(−25%)가 mixed precision보다 효과적. 둘을 합치면 −30%.

---

## Artifacts

| 파일 | 경로 |
|---|---|
| Sensitivity JSON (layer×step) | `results/step_sensitivity_steps10_cal4_seed1000.json` |
| Block sensitivity JSON (block×step) | `results/block_sensitivity_steps10_cal4_seed1000.json` |
| Greedy schedules JSON | `results/greedy_schedules.json` |
| 전체 결과 CSV | `/data/jameskimh/james_dit_pixart_sigma_xl_mjhq/SVDQUANT/MJHQ/sweep_results.csv` |



---
