# DiT Mixed-Precision A3W3 — Experimental Plan

**Model**: PixArt-Sigma-XL-2-1024-MS · **Steps**: 10 · **Eval**: MJHQ-30K (n=100), FID/CLIP/PSNR/SSIM/LPIPS, TPI, MB
**Baseline**: NVFP4 W4A4 (FID 124.1, TPI 4.72s)
**Goal**: A3W3 도달 가능성 검증 + 실패 시 Pareto frontier (W4A3, W3A4) 확장

---

## 1. Method Overview

네 축의 lever를 단계적으로 결합:

| Axis | Lever | Motivation |
|---|---|---|
| **A. Outlier transfer** | SmoothQuant α / AWQ ratio (step-conditional) | adaLN으로 step별 activation 분포 다름 → step-wise α 필요 |
| **B. Weight compensation** | LoRA branch (step-conditional or shared) | Outlier가 weight에 옮겨지면 weight quant 부담 ↑ → low-rank residual로 흡수 |
| **C. Number format** | NVFP4 / FP3 (E2M0, E1M1, E0M2) / INT3 | Heavy-tailed activation에 어떤 format이 최적인지 미검증 |
| **D. Step efficiency** | Group / Skip (DeepCache-style) | 후반 step tolerant → 압축/생략 여지 |

### 핵심 가설

1. **H1**: Step-conditional α가 single α보다 W4A4에서 의미 있게 우월 (FID −5 이상)
2. **H2**: Step-conditional LoRA가 single LoRA 대비 같은 메모리 budget에서 W3A4 회복
3. **H3**: FP3 (특히 E1M1)이 INT3보다 heavy-tailed activation에서 우월
4. **H4**: A3W3은 PTQ-only로는 불가, light LoRA-tuning 필요

---

## 2. Phase Plan

### Phase 0 — Baseline 강화 (1주)
기존 NVFP4에 outlier handling이 있는지 확인 후, 없으면 추가.

| Exp | Config | 목적 |
|---|---|---|
| 0.1 | NVFP4 + SmoothQuant (α=0.5, single) | LLM-style baseline |
| 0.2 | NVFP4 + AWQ | Activation-aware weight selection |
| 0.3 | NVFP4 + QuaRot (Hadamard) | Reviewer가 반드시 묻는 baseline |

**Success**: W4A4 FID 124 → 110 이하

### Phase 1 — Step-conditional outlier handling (1~2주)
α / AWQ ratio를 step별로 분리.

| Exp | Config | 목적 |
|---|---|---|
| 1.1 | Step-uniform α grid search [0.3, 0.5, 0.7, 0.9] | Reference |
| 1.2 | Per-step α (10개 독립 calibration) | Upper bound |
| 1.3 | Step-cluster α (early/mid/late, K=3) | 효율적 중간점 |
| 1.4 | α(t) = MLP(step embedding) | adaLN과 같은 path |

**Metric**: W4A4 / W4A3 / W3A4 각각에서 FID, calibration data 양 (n_calib sweep)

### Phase 2 — Format ablation (1주)
α 고정 후 weight/activation format만 변경.

| Exp | Weight | Activation | 목적 |
|---|---|---|---|
| 2.1 | NVFP4 | INT3 | 기존 W4A3 (FID 358) 회복 |
| 2.2 | NVFP4 | FP3-E2M0 | Integer-like, larger range |
| 2.3 | NVFP4 | FP3-E1M1 | Balanced (likely best) |
| 2.4 | NVFP4 | FP3-E0M2 | Mantissa-heavy, fine resolution near 0 |
| 2.5 | INT3 | NVFP4 | W3A4 reference |
| 2.6 | FP3-E1M1 | NVFP4 | Symmetric FP3 weight |

**Metric**: FID + activation distribution histogram before/after format. Heavy-tail이 잘 표현되는 format 식별.

### Phase 3 — LoRA compensation (1~2주)
Phase 1의 best α + Phase 2의 best format에 LoRA 추가.

| Exp | LoRA config | Memory cost |
|---|---|---|
| 3.1 | Single LoRA, rank ∈ {16, 32, 64, 128} | Baseline |
| 3.2 | Sensitivity-weighted rank (sensitive layer 高 rank) | 같은 budget, 재할당 |
| 3.3 | Step-cluster LoRA (K=3, rank=32 each) | 3× single |
| 3.4 | Step-low-rank: $L^{(t)} = \sum_k \alpha_k(t) U_k$ (K=4) | 4× single + tiny MLP |
| 3.5 | Per-step LoRA (10 separate, rank=16) | 10× rank-16 ≈ single rank-160 |

**Tuning**: PTQ-only → light LoRA fine-tuning (몇 시간) 단계적. PTQ만으로 closure하는지 확인.

### Phase 4 — A3W3 도전 (1~2주)
Phase 1+2+3 best 조합으로 A3W3.

| Exp | Config |
|---|---|
| 4.1 | Best Phase 3 config + activation INT3 |
| 4.2 | Best Phase 3 config + activation FP3-E1M1 |
| 4.3 | + light LoRA tuning (trajectory matching loss, 5k step) |
| 4.4 | + sensitivity-aware schedule (S3-k1 + 위 모두) |

**Decision point**: FID < 200 도달 여부. 미달 시 H4 confirmed → "PTQ ceiling" mechanistic 분석으로 pivot.

### Phase 5 — Step efficiency (1주)
직교 lever. Best A3W3 config 위에 추가.

| Exp | Config |
|---|---|
| 5.1 | DeepCache (interval=2) + Phase 4 best |
| 5.2 | Step skip on tolerant steps (s8, s9 cache) |
| 5.3 | Step group: 10 step → 5 effective (pair averaging) |

**Metric**: TPI 감소 vs FID 손실의 Pareto.

### Phase 6 — Final Pareto + ablation (1주)
모든 lever 조합. Ablation table로 각 component 기여도 분리.

---

## 3. Sensitivity-Guided Allocation

Phase 2 (기존) heatmap 활용:
- **Most tolerant**: attn2_out, late steps (s7–s9) → INT3 OK, low rank LoRA
- **Most sensitive**: mlp_fc2 s1, attn1_qkv s1 → FP3-E1M1 + high rank LoRA, 또는 W4A4 유지
- **Block 27**: s9 가장 tolerant, s0 가장 sensitive → step-conditional 절대 필요

---

## 4. Evaluation Protocol

- **Primary**: FID (n=100 → 최종 n=1000), TPI, Weight MB
- **Secondary**: CLIP, SSIM, LPIPS, PSNR
- **Sanity**: Activation distribution histogram (before/after smoothing)
- **Reference**: FP16 (FID 0 by definition, TPI 0.44s w/ W4FP16)

각 phase 결과는 `results/phase{N}_results.csv`에 누적. 모든 generated image는 `outputs/PHASE{N}_{config}/` 보존.

---

## 5. Risk & Decision Tree

```
Phase 0 fail (FID 동일)
  → NVFP4가 이미 outlier handling 포함 → Phase 1로 직진

Phase 1 fail (step-conditional 효과 없음)
  → adaLN 가설 reject → α 고정하고 Phase 2~3 진행

Phase 4 success (A3W3 FID < 150)
  → 강력한 결과 → NeurIPS main 가능

Phase 4 fail (A3W3 FID > 250 even with tuning)
  → H4 confirmed → "PTQ ceiling for DiT activation" 논문으로 pivot
  → W4FP16 / W3FP16 + step efficiency가 진짜 Pareto라는 결론
```

---

## 6. Deliverables

- **Code**: 모든 quantizer / LoRA / scheduler를 single config로 토글
- **Tables**: Phase별 sweep CSV + final Pareto table
- **Figures**: (1) Sensitivity heatmap × format, (2) Pareto FID vs MB vs TPI 3D, (3) α(t) learned profile
- **Paper angle**: "Step-conditional outlier transfer + LoRA compensation extends DiT PTQ Pareto" (Phase 4 결과에 따라 framing 결정)

---

## 7. Timeline (예상 총 7~9주)

| Week | Phase |
|---|---|
| 1 | Phase 0 |
| 2–3 | Phase 1 |
| 4 | Phase 2 |
| 5–6 | Phase 3 |
| 7–8 | Phase 4 |
| 9 | Phase 5 + 6 |

각 phase 끝에 go/no-go 결정. Phase 4가 main risk point.
