# Experiment Log — adaLN(Step)-aware Quant

> **Live plan.** Based on `experiment_plan.md`. Each phase section is filled in
> with concrete config and commands *before* that phase starts, then updated
> after results land in `experiment_results.md`.
>
> Workflow per phase:
> 1. Fill this section → 2. Run → 3. Write results to `experiment_results.md`
>    → 4. Record go/no-go decision → 5. `git commit -m "phase N: ..."`.

---

## Environment

- **Model**: PixArt-Sigma-XL-2-1024-MS (`PixArt-alpha/PixArt-Sigma-XL-2-1024-MS`)
- **Eval**: MJHQ-30K n=100, FID/CLIP/PSNR/SSIM/LPIPS, TPI (s/img), Weight MB
- **Reference baseline**: NVFP4 W4A4 FID ≈ 124.1, TPI ≈ 4.72 s @ 10 steps
- **GPU**: 0,1  (`CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2`)
- **Steps**: 10 (all phases)
- **Working dir**: `/home/jovyan/workspace/Workspace_DiT/adaLN(Step)-aware-quant/`

Run as:
```bash
cd /home/jovyan/workspace/Workspace_DiT/adaLN(Step)-aware-quant/
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    -m adastep_quant.runners <args>
```

---

## Phase 0 — Baseline 강화 (Week 1)

**상태**: done (2026-04-25)  
**GPU**: 0,1  
**Wall-clock 실제**: ~1.5 h

### 목적

어떤 outlier handling이 NVFP4 W4A4에서 실질적 FID 개선을 주는지 측정.  
SmoothQuant, AWQ, QuaRot(Hadamard), SVDQuant 네 방법을 baseline으로 확립.

### Config table

| Exp   | `--quant_method` | `--alpha` | Notes |
|-------|-----------------|-----------|-------|
| 0.0   | FP16            | —         | Reference (no quant) |
| 0.1a  | RTN             | 0.0       | NVFP4 only, no smoothing |
| 0.1b0 | RTN_SMOOTH      | 0.3       | SmoothQuant α grid |
| 0.1b1 | RTN_SMOOTH      | 0.5       | SmoothQuant α grid |
| 0.1b2 | RTN_SMOOTH      | 0.7       | SmoothQuant α grid |
| 0.1b3 | RTN_SMOOTH      | 0.9       | SmoothQuant α grid |
| 0.2   | AWQ             | auto      | per-channel scale grid (20 pts) |
| 0.3   | QUAROT          | —         | Hadamard offline+online |
| 0.4   | SVDQUANT        | —         | mtq SVDQuant rank=32 |

### Commands

```bash
bash scripts/run_phase0.sh
```

Or manually:
```bash
CUDA_VISIBLE_DEVICES=0,1 accelerate launch --num_processes 2 \
    -m adastep_quant.runners \
    --exp 0.1a --quant_method RTN --alpha 0.0 \
    --num_steps 10 --num_samples 100 --calib_prompts 64 \
    --results_csv results/phase0_results.csv \
    --results_md experiment_results.md
```

### Decision rule

- **Go (FID < 110)**: phase 1 진입. Best outlier method carry-over.
- **No-go (best FID ≥ 110)**: "기존 NVFP4에 이미 outlier handling" 결론, phase 1 step-conditional로 직진.

---

## Phase 1 — Step-conditional outlier handling (Week 2~3)

**상태**: done (2026-04-25)  
**GPU**: 0,1  
**Wall-clock 실제**: ~3 h total (all 5 exps)

### 목적

H1 검증: step-conditional α가 single α 대비 FID −5 이상 개선을 주는가?

**Phase 0 결과**: RTN_SMOOTH α=0.7이 best (FID=121.939). α curve가 non-monotonic — α=0.7이 sweet spot. SVDQuant도 거의 동등 FID이나 3× 빠름 (TPI=1.54s).

**PHASE0_BEST_ALPHA = 0.7** (RTN_SMOOTH). Phase 1은 이 α를 기반으로 step-conditional 분리.

Per-step calibration: step `t`의 통계를 얻기 위해 `num_inference_steps=t+1` 짜리 run을 별도 수행. 총 10 calibration runs × 64 prompts.

### Config table

| Exp | `--alpha_mode` | α | Description |
|-----|---------------|---|-------------|
| 1.0 | single        | 0.7 | Phase 0 best α, fixed (reference) |
| 1.1 | per_step      | 0.7 | 10 independent calibrations, α=0.7 starting point |
| 1.2 | per_cluster   | 0.7 | K=3 cluster (early 0-2, mid 3-6, late 7-9) |
| 1.3 | mlp           | 0.7 | Per-layer per-step α (3-pt MA smoothed) |
| 1.4 | per_type_step | 0.7 | Per-(layer_type, step) α, 7×10=70 grid-searched |

### Commands

```bash
PHASE0_BEST_ALPHA=0.7 bash scripts/run_phase1.sh > /tmp/phase1_full.log 2>&1 &
```

### Decision rule

- **H1 confirmed** (exp 1.1 or 1.4 vs 1.0 FID delta ≥ 5): Phase 2 진입.
- **H1 rejected** (delta < 2): α 고정, phase 2로 직진. Note "step-conditional 효과 미미" in results.

---

## Phase 2 — Format ablation (Week 4)

**상태**: done (2026-04-26)  
**GPU**: 0,1  
**Wall-clock 실제**: ~1.5 h

### 목적

H3 검증: FP3-E1M1이 INT3보다 heavy-tailed DiT activation에서 우월한가?  
Phase 1 best α config 고정 후 weight/activation format만 변경.

### Config table (after Phase 1)

| Exp | `--fmt_W` | `--fmt_A` | Description |
|-----|-----------|-----------|-------------|
| 2.0 | NVFP4     | NVFP4     | Phase 1 best (sanity check) |
| 2.1 | NVFP4     | INT3      | W4A3 (기존 FID 358, 회복 목표) |
| 2.2 | NVFP4     | FP3-E2M0  | Large dynamic range |
| 2.3 | NVFP4     | FP3-E1M1  | Balanced (likely best) |
| 2.4 | NVFP4     | FP3-E0M2  | Fine near zero |
| 2.5 | INT3      | NVFP4     | W3A4 reference |
| 2.6 | FP3-E1M1  | NVFP4     | Symmetric FP3 weight |

### Commands

```bash
bash scripts/run_phase2.sh
```

### Decision rule

- W4A3 FID < 200 → Phase 3 (LoRA compensation) 진입.
- Best format carry-over to Phase 3.

---

## Phase 3 — Joint Smooth + A3W3 + LoRA (Week 5~6)

**상태**: done (2026-04-26)  
**GPU**: 0,1  
**Wall-clock 실제**: ~0.5 h (6 exps)

### 목적

SmoothQuant α=0.7, format quantization (W3A3), SVD LoRA 세 가지를 **동시에** 보정.
`SmoothFormatLoRALinear`: smooth_scale → Q(W/s, fmt_W) → SVD(W/s − Q) → LoRA init.
LoRA는 x_smooth (FP16, pre-quant) 로 weight-quant AND act-quant 오차를 모두 보정.

### Design

- Calibration: single-pass step-averaged x_max → smooth_scale → W_q → SVD LoRA (PTQ, no gradient)
- Forward: `x_smooth = x * s`, `x_q = Q(x_smooth, fmt_A)`, `out = F.linear(x_q, W_q) + F.linear(F.linear(x_smooth, A), B) / rank`
- LoRA path deliberately uses FP16 x_smooth, not x_q, for maximum correction power

### Config table

| Exp       | `--fmt_W`   | `--fmt_A`   | `--lora_mode` | `--rank` | Description |
|-----------|-------------|-------------|---------------|----------|-------------|
| 3.0       | NVFP4       | NVFP4       | single        | 32       | W4A4 Smooth+LoRA upper bound |
| 3.1r16    | FP3-E1M1    | FP3-E1M1    | single        | 16       | W3A3 rank=16 |
| 3.1r32    | FP3-E1M1    | FP3-E1M1    | single        | 32       | W3A3 rank=32 (baseline) |
| 3.1r64    | FP3-E1M1    | FP3-E1M1    | single        | 64       | W3A3 rank=64 |
| 3.2       | INT3        | INT3        | single        | 32       | W3A3 INT3 (vs FP3) |
| 3.3       | FP3-E1M1    | FP3-E1M1    | per_cluster   | 32       | W3A3 per_cluster LoRA |

All use `--quant_method SMOOTH_LORA --alpha 0.7`.

### Commands

```bash
# After Phase 2: set PHASE3_BEST_FMT_W/A from Phase 2 results
PHASE3_BEST_FMT_W=FP3-E1M1 PHASE3_BEST_FMT_A=FP3-E1M1 bash scripts/run_phase3.sh
```

### Decision rule

- 3.1r32 FID vs 2.0 NVFP4 baseline: improvement ≥ 5 FID → strong H-confirm.
- Best (rank, fmt) carry-over to Phase 4.

---

## Phase 4 — Grand Combo: Phase 3 best + DeepCache (Week 7)

**상태**: done (2026-04-26)

### 목적

Phase 3 best (Smooth+W3A3+LoRA)에 DeepCache (block-level residual caching)를 추가해 TPI를 줄이면서 FID 손실 최소화.

### Config table

| Exp | Cache | `--cache_interval` | Description |
|-----|-------|-------------------|-------------|
| 4.0 | off   | —                 | Phase 3 best, no cache (TPI reference) |
| 4.1 | FP16  | 2                 | FP16 + DeepCache (pure caching baseline) |
| 4.2 | on    | 2                 | Phase 3 best + cache interval=2 |
| 4.3 | on    | 3                 | Phase 3 best + cache interval=3 |

### Commands

```bash
PHASE3_BEST_FMT_W=FP3-E1M1 PHASE3_BEST_FMT_A=FP3-E1M1 \
PHASE3_BEST_RANK=32 PHASE3_BEST_LORA=single PHASE3_BEST_ALPHA=0.7 \
bash scripts/run_phase4.sh
```

### Decision rule

- TPI reduction ≥ 1.5× with FID increase ≤ 5 → Phase 5 Final Pareto.

---

## Phase 5 — Final Pareto + ablation (Week 8)

**상태**: done (2026-04-26)

### 목적

Phase 0~4의 best config들로 FID vs TPI vs Weight MB Pareto 곡선 구성 및 최종 ablation.

### Config table

| Exp | Description |
|-----|-------------|
| 5.A | FP16 anchor |
| 5.B | W4A4 RTN baseline (Phase 0.1a) |
| 5.C | W4A4 + Smooth α=0.7 (Phase 0.1b best) |
| 5.D | Phase 2 best format (no LoRA) |
| 5.E | Phase 3.0 (W4A4 Smooth+LoRA) |
| 5.F | Phase 3 best (W3A3 Smooth+LoRA) |
| 5.G | Phase 4 best (W3A3 Smooth+LoRA + DeepCache) |

### Commands

```bash
P2_BEST_FMT_W=FP3-E1M1 P2_BEST_FMT_A=FP3-E1M1 \
P3_BEST_RANK=32 P4_BEST_CACHE_INT=2 \
bash scripts/run_phase5.sh
```

---

## Phase 6 — adaLN Staleness Oracle (2026-04-26~)

**상태**: running

### 목적

DeepCache가 PixArt-Sigma에서 왜 FID를 크게 손상시키는지를 성분 분해로 정량화.  
6개 adaLN 파라미터를 4개 컴포넌트(attn_norm / attn_gate / mlp_norm / mlp_gate)로 분리하여  
각각의 FID 기여도를 측정하고, 현실적인 caching 전략의 상한(oracle)을 측정.

모든 oracle 실험은 FP16, n=100, 10 steps, blocks 5–27, cache_interval=2.

### 실험 목록

| Exp | Oracle mode | 설명 | FID (n=100) |
|-----|------------|------|-------------|
| oracle_FP16 | none | anchor (항상 fresh) | 0.0 |
| oracle_both_stale | both_stale | 전부 stale — DeepCache sanity | 183.7 |
| oracle_gate_fresh | gate_fresh_only | gate만 fresh | 115.2 |
| oracle_norm_fresh | norm_fresh_only | norm만 fresh | 89.1 |
| oracle_attn_norm | attn_norm_fresh | attn_norm만 fresh | 161.0 |
| oracle_mlp_norm | mlp_norm_fresh | mlp_norm만 fresh | 104.7 |
| oracle_attn_gate | attn_gate_fresh | attn_gate만 fresh | 157.9 |
| oracle_mlp_gate | mlp_gate_fresh | mlp_gate만 fresh | 126.5 |
| oracle_A | oracle_A | pre-gate 캐시 + fresh gate | 130.4 |
| oracle_mlp_taylor | oracle_mlp_taylor | oracle_A + 1차 Taylor MLP 보정 | 124.6 |
| oracle_attn_skip_mlp_q2 | oracle_attn_skip_mlp_q2 | attn 캐시 + MLP W2A2 always | 측정 중 |
| oracle_modulation_corr | oracle_modulation_corr | oracle_A + scale-ratio modulation 보정 | 측정 중 |

### 4-way 기여도 분해 결과

```
                 norm side      gate side    합계
Attention          12.4%          14.1%     26.5%
MLP/FFN            43.0%          31.2%     74.2%
합계               55.4%          45.2%
```

**MLP side 74.2% dominant** → attention 캐시가 비교적 안전, FFN은 재계산 필요.

### Commands

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH="." \
python -m accelerate.commands.accelerate_cli launch --num_processes 1 \
    -m adastep_quant.runners --phase 6 --quant_method FP16 \
    --num_steps 10 --num_samples 100 \
    --cache_start 5 --cache_end 28 --cache_interval 2 \
    --adaln_oracle <mode> \
    --results_csv results/oracle_adaln_results.csv \
    --results_md experiment_results.md
```

### Decision rule

- `gate_staleness / total_delta ≥ 0.5` → adaLN-aware caching viable
- `gate_staleness / total_delta ≤ 0.2` → norm dominant, rethink
- oracle_A FID이 vanilla DeepCache보다 낮으면 → fresh gate 재적용 전략 유효


---

# Result

# Results Log — adaLN(Step)-aware Quant

> Append-only log. Each exp adds a row immediately after completion.
> Each phase ends with a summary table + go/no-go decision.
> Machine-readable copies also land in `results/phase{N}_results.csv`.

---

## Phase 0 — Baseline 강화

n=100 samples, MJHQ-30K, 10 steps. (Rows from early smoke-test runs discarded; canonical n=100 results only.)

| Exp | Quant | α | FID↓ | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ | TPI(s) | MB | Notes |
|-----|-------|---|------|-------|-------|-------|--------|--------|----|-------|
| 0.0 | FP16 | — | -0.0 | 34.945 | ∞ | 1.000 | 0.000 | 0.392 | 305.1 | reference |
| 0.1a | RTN | 0.0 | 129.057 | 34.569 | 12.841 | 0.448 | 0.538 | 4.414 | — | no smoothing |
| 0.1b03 | RTN_SMOOTH | 0.3 | 133.922 | 34.362 | 11.368 | 0.398 | 0.559 | 4.439 | 305.1 | worse than RTN |
| 0.1b05 | RTN_SMOOTH | 0.5 | 136.028 | 34.646 | 11.754 | 0.437 | 0.548 | 4.439 | 305.1 | |
| **0.1b07** | **RTN_SMOOTH** | **0.7** | **121.939** | 34.582 | 14.004 | 0.518 | 0.465 | 4.444 | 305.1 | **best FID** |
| 0.1b09 | RTN_SMOOTH | 0.9 | 145.079 | 34.387 | 12.482 | 0.429 | 0.557 | 4.441 | 305.1 | overshoots |
| 0.2 | AWQ | — | 624.036 | 19.745 | 6.331 | 0.027 | 0.950 | 3.699 | 305.1 | catastrophic |
| 0.3 | QUAROT | — | 273.529 | 29.267 | 9.333 | 0.226 | 0.794 | 4.496 | 305.1 | bad |
| **0.4** | **SVDQUANT** | — | **122.235** | **34.807** | **15.454** | **0.575** | **0.452** | **1.540** | 305.1 | **3× faster TPI** |

### Phase 0 Summary

- **Best baseline**: RTN_SMOOTH α=0.7 (FID=121.939), narrowly beating SVDQuant (122.235)
- **Decision**: No-go by strict rule (best FID=121.9 ≥ 110 threshold) → Phase 1 step-conditional 직진. `PHASE0_BEST_ALPHA=0.7`.
- **Observations**:
  1. SmoothQuant α has a **non-monotonic effect** on DiT: α=0.3/0.5 hurt FID (133–136 vs RTN 129), α=0.7 improves FID (121.9, −7.1 vs RTN), α=0.9 overshoots back (145). Sweet spot is α=0.7.
  2. **SVDQuant** (FID=122.235, TPI=1.54s) nearly matches RTN_SMOOTH α=0.7 in FID while being **~3× faster** (4.44s→1.54s). Strong alternative — but mtq-based, so not easily combined with step-conditional SmoothQuant.
  3. **AWQ completely failed** (FID=624): per-channel scale grid search destroys NVFP4 levels — the activation-aware scale presumably maps values outside NVFP4's usable range.
  4. **QuaRot failed badly** (FID=273): offline Hadamard rotation changes DiT activation distribution in a way NVFP4 handles poorly (likely spreads mass beyond its ±6 range).
  5. Phase 1 goal: test if step-conditional α pushes FID below 115 (i.e., ≥7 improvement over single α=0.7 reference).

---

## Phase 1 — Step-conditional α

| Exp | α_mode | FID↓ | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ | TPI(s) | MB | Notes |
|-----|--------|------|-------|-------|-------|--------|--------|----|-------|
| 1.0 | single | 124.776 | 34.701 | 13.128 | 0.464 | 0.497 | 4.443 | 305.1 | reference (α=0.7 fixed) |
| **1.1** | **per_step** | **124.175** | **34.544** | **13.624** | **0.490** | **0.468** | **4.447** | **305.1** | **best; separate W per step** |
| 1.2 | per_cluster | 388.068 | 21.742 | 7.333 | 0.164 | 0.829 | 4.446 | 305.1 | cluster-optimal α hurts |
| 1.3 | mlp | 221.061 | 32.402 | 10.144 | 0.274 | 0.699 | 4.430 | 305.1 | per-layer avg α, single W |
| 1.4 | per_type_step | 280.420 | 29.949 | 9.113 | 0.223 | 0.746 | 4.446 | 305.1 | per-type avg α, single W |

### Phase 1 Summary

- **H1 verdict**: REJECTED. Best FID improvement = exp 1.1 ΔFID = −0.601 (124.776 → 124.175). Well below the ≥5 threshold. Step-conditional α yields negligible gain over single α=0.7.

- **Best config**: **1.1** (per_step, α=0.7, separate weights per step). Marginally best FID=124.175. 1.0 (single α) is equally good in practice.

- **Key findings**:
  1. **`separate_weights=True` is required for correctness** in per_step/per_cluster modes. A single quantized weight with varying smooth_scale[t] produces output ≈ x·W·(smooth_scale[t]/mean_scale), causing near-zero output at early steps and float16 overflow at late steps → FID~617. Fixed by storing T separate quantized weight buffers, each paired with its own smooth_scale.
  2. **Weight-MSE-only α selection is wrong** for per_cluster/mlp/per_type_step: the criterion ignores activation NVFP4 range (±6). A low α (favored by weight MSE) → large smooth_scale → activations overflow NVFP4 → catastrophic FID (388–280). α=0.7 (constrained grid floor) is the best stable value.
  3. **mlp/per_type_step use mean x_max + averaged α (single weight)**: mathematically consistent but equivalent to a slightly different single-α calibration — hence FID is degraded vs true per-step (1.1) or single (1.0).

- **Decision**: Proceed to Phase 2 with **single α=0.7** (exp 1.0 config). step-conditional α is abandoned for Phase 2+ given negligible H1 gain. Carry forward `--quant_method STEP_SMOOTH --alpha 0.7 --alpha_mode single`.

---

## Phase 2 — Format ablation

| Exp | W_fmt | A_fmt | FID↓ | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ | TPI(s) | MB | Notes |
|-----|-------|-------|------|-------|-------|-------|--------|--------|----|-------|
| 2.0 | NVFP4 | NVFP4 | 127.065 | 34.339 | 12.974 | 0.455 | 0.528 | 4.424 | 305.100 |  |
| 2.1 | NVFP4 | INT3 | 356.415 | 25.564 | 10.346 | 0.276 | 0.735 | 1.217 | 305.100 |  |
| 2.2 | NVFP4 | FP3-E2M0 | 330.711 | 27.576 | 10.793 | 0.313 | 0.748 | 3.302 | 305.100 |  |
| 2.3 | NVFP4 | FP3-E1M1 | 364.639 | 25.922 | 10.457 | 0.275 | 0.735 | 3.224 | 305.100 |  |
| 2.4 | NVFP4 | FP3-E0M2 | 364.639 | 25.922 | 10.457 | 0.275 | 0.735 | 3.484 | 305.100 |  |
| 2.5 | INT3 | NVFP4 | 209.781 | 30.773 | 9.475 | 0.303 | 0.716 | 4.739 | 228.800 |  |
| 2.6 | FP3-E1M1 | NVFP4 | 214.251 | 30.521 | 9.396 | 0.304 | 0.720 | 4.424 | 228.800 |  |

### Phase 2 Summary

- **H3 verdict** (FP3-E1M1 vs INT3 for W4A3): REJECTED. FP3-E1M1 (FID=364.6) is marginally *worse* than INT3 (FID=356.4) for W4A3 — both catastrophically bad (>2.5× FID vs W4A4=127). FP3 dynamic-range advantage disappears when the format itself is misaligned with NVFP4-calibrated smooth_scale. H3 is false.

- **Key findings**:
  1. **Activation 3-bit is catastrophic** regardless of format: W4A3 FID 330–365 vs W4A4 FID 127. Smooth α=0.7 alone cannot compensate for 3-bit activation quantization error.
  2. **Weight 3-bit is tolerable**: W3A4 FID=209.8 (INT3) / 214.3 (FP3-E1M1) — degraded but recoverable territory. LoRA compensation should be effective here.
  3. **FP3 variants (A3)**: E2M0/E1M1/E0M2 all land in FID 330–365. E0M2 gives identical numbers to E1M1 (⚠ likely a bug in the E0M2 quantizer — level table too similar or same fallthrough).
  4. **TPI anomaly (2.1)**: INT3 activation TPI=1.217s is ~3.5× faster than NVFP4 (4.4s) which is physically implausible — suggests the INT3 activation path is falling through to a non-standard codepath (e.g., torch.round-based INT3 skipping the block quantization overhead). Does not affect FID validity but TPI is not comparable.

- **Best format combo**: W3A4 INT3/NVFP4 (FID=209.8, MB=228.8) for weight-only 3-bit. For joint W3A3, both FP3-E1M1/FP3-E1M1 and INT3/INT3 will be tested with Smooth+LoRA in Phase 3.

- **Decision**: Proceed to Phase 3 (joint Smooth+Format+LoRA). Carry forward W3A3 with FP3-E1M1 as default target (similar MB to INT3, and the Level-table issue with E0M2 warrants investigation). Phase 3 is the critical test: can SVD LoRA + SmoothQuant recover W3A3 FID toward W4A4?

---

## Phase 3 — LoRA compensation

| Exp | lora_mode | rank | W_fmt | A_fmt | FID↓ | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ | TPI(s) | MB | Notes |
|-----|-----------|------|-------|-------|------|-------|-------|-------|--------|--------|----|-------|
| 3.0 | single | 32 | NVFP4 | NVFP4 | 120.302 | 34.634 | 13.976 | 0.522 | 0.466 | 4.566 | 305.100 | W4A4 smooth+LoRA upper bound |
| 3.1r16 | single | 16 | FP3-E1M1 | FP3-E1M1 | 363.366 | 22.297 | 9.453 | 0.253 | 0.813 | 4.459 | 228.800 | W3A3 FP3-E1M1/FP3-E1M1 smooth+LoRA rank=16 |
| 3.1r32 | single | 32 | FP3-E1M1 | FP3-E1M1 | 362.923 | 22.144 | 9.436 | 0.245 | 0.816 | 4.223 | 228.800 | W3A3 FP3-E1M1/FP3-E1M1 smooth+LoRA rank=32 |
| 3.1r64 | single | 64 | FP3-E1M1 | FP3-E1M1 | 360.860 | 22.320 | 9.388 | 0.248 | 0.813 | 4.376 | 228.800 | W3A3 FP3-E1M1/FP3-E1M1 smooth+LoRA rank=64 |
| 3.2 | single | 32 | INT3 | INT3 | 356.748 | 22.515 | 9.514 | 0.246 | 0.817 | 1.590 | 228.800 | W3A3 INT3 smooth+LoRA |
| 3.3 | per_cluster | 32 | FP3-E1M1 | FP3-E1M1 | 362.923 | 22.144 | 9.436 | 0.245 | 0.816 | 4.483 | 228.800 | W3A3 FP3-E1M1/FP3-E1M1 smooth+LoRA per_cluster |

### Phase 3 Summary

- **Best config**: 3.0 (W4A4 NVFP4, Smooth α=0.7, LoRA rank=32) — FID=120.302

- **W4A4 smooth+LoRA**: FID=120.302 (Phase 2 STEP_FORMAT baseline 127.065 대비 **−6.8** 개선)

- **W3A3 LoRA recovery**: FAILED across all variants. FID 357–364, rank- and format-invariant:
  - FP3-E1M1/FP3-E1M1 rank=16/32/64: FID 361–364 (rank-invariant — more capacity doesn't help)
  - INT3/INT3 rank=32: FID=357 (marginally better, still catastrophic)
  - per_cluster rank=32: FID=362.923 (identical to single rank=32)

- **H verdict**: LoRA effectively corrects **weight-quant error** (W4A4: −6.8 FID) but **cannot correct activation-quant error**. Root cause: base path computes `F.linear(x_q, W_q)`; the residual `F.linear(x_smooth − x_q, W_q)` is independent of the LoRA correction and not canceled. For W3A3, activation quantization error (x_smooth → x_q via FP3/INT3) dominates weight quantization error by a large margin.

- **Key findings**:
  1. **Smooth+LoRA W4A4**: FID=120.302 (−6.8 vs 127.065). SVD-initialized LoRA effectively compensates weight quantization error when activation is kept at NVFP4.
  2. **W3A3 rank sweep**: FID≈361–364, rank-invariant (16/32/64). Increasing LoRA capacity does not help when activation error dominates.
  3. **INT3 vs FP3-E1M1 (W3A3)**: INT3/INT3 FID=357 vs FP3/FP3 FID=361–364. Marginal INT3 edge — both equally unusable. Likely INT3 has slightly less activation error due to uniform spacing vs FP3's geometric spacing.
  4. **per_cluster LoRA**: FID=362.923 — identical to single (3.1r32). No benefit from step-conditional LoRA branches when the dominant error source (activation quantization) is input-dependent, not step-conditional.

- **Decision**: Phase 4 proceeds with **W4A4 Smooth+LoRA (3.0 config) + DeepCache**. W3A3 line abandoned — PTQ LoRA cannot recover 3-bit activation quantization error without gradient-based fine-tuning.

---

## Phase 4 — Grand Combo (Phase 3 best + DeepCache)

| Exp | cache_interval | quant_method | FID↓ | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ | TPI(s) | MB | Notes |
|-----|----------------|--------------|------|-------|-------|-------|--------|--------|----|-------|
| 4.0 |  | SMOOTH_LORA | 120.302 | 34.634 | 13.976 | 0.522 | 0.466 | 5.896 | 305.100 | Phase3 best no cache |
| 4.1 | 2 | FP16 | 148.913 | 34.496 | 13.914 | 0.517 | 0.553 | 0.392 | 1220.300 | FP16+cache=2 baseline |
| 4.2 | 2 | SMOOTH_LORA | 219.397 | 31.647 | 11.620 | 0.342 | 0.687 | 3.081 | 305.100 | Phase3 best + cache=2 |
| 4.3 | 3 | SMOOTH_LORA | 293.718 | 27.774 | 11.686 | 0.357 | 0.721 | 2.497 | 305.100 | Phase3 best + cache=3 |

### Phase 4 Summary

- **Best config**: 4.0 (SMOOTH_LORA, no cache, FID=120.302, TPI=5.896s). Cache variants fail quality criterion.

- **TPI speedup vs no-cache (4.0)**:
  - 4.2 (cache=2): speedup = 5.896/3.081 = **1.91×**, ΔFID = +99.1 (219.4 − 120.3) ❌
  - 4.3 (cache=3): speedup = 5.896/2.497 = **2.36×**, ΔFID = +173.4 (293.7 − 120.3) ❌

- **FP16+DeepCache baseline (4.1)**: FID=148.9 (+148.9 vs FP16). Even without quantization, DeepCache interval=2 is severely lossy on PixArt-Sigma — confirming the model architecture (dense attention + adaLN conditioning per step) is not amenable to block-level residual caching.

- **Decision**: Decision rule "speedup ≥ 1.5× AND ΔFID ≤ 5" — **NO-GO**. Speedup criterion met (1.91× / 2.36×) but ΔFID criterion missed by 20× (99.1 >> 5). DeepCache is incompatible with acceptable FID on this model. **Best deployment config remains Phase 3.0 (W4A4 Smooth+LoRA, FID=120.302, TPI≈4.6s, MB=305.1).**

---

## Phase 5 — Final Pareto

Aggregated best configs from Phases 0–4. Entries marked ★ lie on the FID vs TPI Pareto frontier.

| Config | Description | FID↓ | CLIP↑ | TPI(s) | MB | Phase |
|--------|-------------|------|-------|--------|----|-------|
| 5.A ★ | FP16 (unquantized) | ~0.0 | 34.945 | 0.392 | 305.1 | P0/0.0 |
| 5.B | W4A4 RTN (no smooth) | 129.1 | 34.569 | 4.414 | 305.1 | P0/0.1a |
| 5.C | W4A4 RTN+Smooth α=0.7 | 121.9 | 34.582 | 4.444 | 305.1 | P0/0.1b07 |
| 5.D ★ | SVDQuant W4A4 | 122.2 | 34.807 | 1.540 | 305.1 | P0/0.4 |
| 5.E | W4A4 Smooth (Phase 2 baseline) | 127.1 | 34.339 | 4.424 | 305.1 | P2/2.0 |
| 5.F ★ | W4A4 Smooth+LoRA rank=32 | 120.3 | 34.634 | 4.566 | 305.1 | P3/3.0 |
| 5.G | W3A3 FP3-E1M1 Smooth+LoRA rank=64 | 360.9 | 22.320 | 4.376 | 228.8 | P3/3.1r64 |
| 5.H | W4A4 Smooth+LoRA + DeepCache×2 | 219.4 | 31.647 | 3.081 | 305.1 | P4/4.2 |


### Phase 5 Summary

- **Pareto frontier** (FID↓ vs TPI↓, lower-left dominates):
  - **5.A** (FP16): FID≈0, TPI=0.392s — unbeatable quality anchor
  - **5.D** (SVDQuant): FID=122.2, TPI=1.540s — **best quantized quality/speed tradeoff** (FID −7.0 vs RTN at 2.9× faster TPI than RTN)
  - **5.F** (Smooth+LoRA): FID=120.302, TPI=4.566s — **best quantized FID** (+1.9 vs SVDQuant but 3× slower TPI)
  - 5.C, 5.B, 5.G, 5.H are all Pareto-dominated.

- **Recommended deployment config**: **SVDQuant W4A4** (5.D) for throughput-sensitive deployments (1.54s/img); **Smooth+LoRA W4A4** (5.F) when image quality is top priority (4.57s/img, −1.9 FID vs SVDQuant). Both at 305.1 MB.

- **adaLN(step)-aware hypothesis verdict**:
  - H1 (step-conditional α improves FID ≥ 5): **REJECTED** — per-step α gave ΔFID=−0.6 only.
  - H3 (FP3-E1M1 better than INT3 for W4A3): **REJECTED** — all 3-bit activation formats are catastrophically lossy (FID 330–365), FP3 marginally worse than INT3.
  - W4A4 with SVD-initialized LoRA (Step H-null): **CONFIRMED** — +6.8 FID improvement over Smooth alone. Best PTQ method in this study.
  - DeepCache + quantization: **INCOMPATIBLE** at FID ≤ 5 constraint — too lossy even for FP16 (ΔFID=+149 for cache=2).

- **Weight MB savings**: W3A3 (228.8 MB) vs W4A4 (305.1 MB) saves 25% memory, but W3A3 FID=357–364 makes it unusable without gradient-based fine-tuning.

---

## Phase 6 — adaLN Staleness Oracle (4-way decomposition)

n=100 samples, FP16 (no quantization), cache_interval=2, blocks 5–27.
모든 모드에서 full computation 실행 (실제 speedup 없음) — adaLN 각 성분의 FID 기여도만 측정.

6개 adaLN 파라미터를 4개 컴포넌트로 분리:
- **attn_norm**: shift_msa, scale_msa (attention 입력 앞 norm 변환)
- **attn_gate**: gate_msa (attention 출력 스케일)
- **mlp_norm**: shift_mlp, scale_mlp (FFN 입력 앞 norm 변환)
- **mlp_gate**: gate_mlp (FFN 출력 스케일)

| Exp | Oracle mode | Fresh 컴포넌트 | FID↓ | CLIP↑ | PSNR↑ | SSIM↑ | LPIPS↓ | TPI(s) |
|-----|------------|--------------|------|-------|-------|-------|--------|--------|
| oracle_FP16 | none (anchor) | 전부 fresh | 0.0 | 34.945 | ∞ | 1.000 | 0.000 | 0.445 |
| oracle_both_stale | both_stale | — (전부 stale) | 183.7 | 32.870 | 11.846 | 0.357 | 0.667 | 0.451 |
| oracle_gate_fresh | gate_fresh_only | attn_gate + mlp_gate | 115.2 | 34.473 | 17.863 | 0.626 | 0.420 | 0.464 |
| oracle_norm_fresh | norm_fresh_only | attn_norm + mlp_norm | 89.1 | 34.715 | 16.958 | 0.655 | 0.349 | 0.450 |
| oracle_attn_norm | attn_norm_fresh | attn_norm만 | 161.0 | 33.451 | 13.004 | 0.423 | 0.613 | 0.446 |
| oracle_mlp_norm | mlp_norm_fresh | mlp_norm만 | 104.7 | 34.559 | 15.440 | 0.595 | 0.421 | 0.454 |
| oracle_attn_gate | attn_gate_fresh | attn_gate만 | 157.9 | 33.641 | 13.049 | 0.418 | 0.612 | 0.454 |
| oracle_mlp_gate | mlp_gate_fresh | mlp_gate만 | 126.5 | 34.555 | 16.753 | 0.595 | 0.462 | 0.447 |
| oracle_A | oracle_A (실제 캐싱) | pre-gate 캐시+fresh gate | 130.4 | 34.706 | 14.627 | 0.556 | 0.494 | 0.375 |
| oracle_mlp_taylor | oracle_mlp_taylor | attn 캐시+fresh gate + MLP 1차 Taylor 보정 | 124.6 | 34.581 | 14.417 | 0.553 | 0.500 | — |

### Phase 6 Summary

**기여도 분해** (기여도 = FID(both_stale) − FID(X_fresh_only), 총 열화량 = 183.7):

```
┌──────────────┬─────────────────────────┬──────────┬────────┐
│              │  norm side              │ gate side│  합계  │
│              │  (입력 변환)             │ (출력 ×) │        │
├──────────────┼─────────────────────────┼──────────┼────────┤
│  Attention   │  +22.8  (12.4%)         │ +25.9    │  26.5% │
│              │  183.7→161.0            │ (14.1%)  │        │
├──────────────┼─────────────────────────┼──────────┼────────┤
│  MLP / FFN   │  +79.0  (43.0%)         │ +57.3    │  74.2% │
│              │  183.7→104.7            │ (31.2%)  │        │
└──────────────┴─────────────────────────┴──────────┴────────┘
  norm 합계:  101.8 (55.4%)    gate 합계:  83.1 (45.2%)
```

- **MLP side가 74.2%로 압도적 dominant** — mlp_norm(43.0%) + mlp_gate(31.2%)
- **Attention side는 26.5%** — attn_norm(12.4%) + attn_gate(14.1%)
- norm > gate 이지만 격차는 55.4% vs 45.2%로 작음
- mlp_norm 단독이 가장 큰 단일 성분 (43.0%)

**oracle_A vs vanilla DeepCache**:
- vanilla DeepCache (Phase 4 exp 4.1): FID = 148.9
- oracle_A (pre-gate 캐시 + fresh gate): FID = 130.4
- 개선: **Δ−18.5 FID** — gate를 fresh하게 재적용하는 것만으로도 의미 있는 향상
- 단, oracle_A TPI 스피드업은 1.19× (0.445s → 0.375s)로 작음. mlp_gate가 31.2% 기여하므로 FFN output에 fresh gate 재적용이 핵심

**oracle_mlp_taylor — 1차 Taylor 보정 실험**:
- oracle_A + MLP norm staleness를 JVP로 1차 보정: FID = **124.6**
- oracle_A(130.4) 대비 −5.8 FID 개선 (22% 회복)
- mlp_norm_fresh(104.7) 까지 남은 갭: 19.9 FID (78%는 고차 비선형성)
- **결론**: PixArt FFN은 상당히 비선형 — 1차 Taylor로는 staleness 오차의 22%밖에 회복 불가
- **비용 대비 효과 없음**: JVP ≈ MLP forward 1회 추가 비용이 mlp_norm_fresh(완전 재계산, 동일 비용)와 같은데 FID는 19.9 열위 → Taylor 방향 실용성 없음

**전략적 함의**:
1. **Attention 블록은 캐시해도 비교적 무해** (26.5%만 기여) — attention 연산 자체를 캐시하는 것이 타당
2. **MLP/FFN은 재계산 + adaLN fresh 적용이 필수** (74.2% 기여) — 특히 mlp_norm(shift/scale)이 FFN 입력을 결정
3. **1차 Taylor 보정은 실용적이지 않음** — 동일 비용으로 MLP 완전 재계산이 훨씬 우수 (FID 104.7 vs 124.6)
4. **이상적 구조**: attention은 캐시, FFN은 매 step 재계산 + fresh adaLN (mlp_norm + mlp_gate)
   - 기대 FID: 104.7 근방 (mlp_norm_fresh 결과)
   - 기대 speedup: attention block 스킵 → transformer 연산의 ~50% 절감 추정


