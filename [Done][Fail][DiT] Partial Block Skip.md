### Partial block skip

* block 전체를 skip하지 말고 block 내부의 attention만 또는 MLP만 skip. 

* DiT block 내부 연산 비율 (1024px, 28 blocks, 대략적 추정):
    * Self-attention: ~55–60% (sequence length 4096 정도에서 quadratic term dominant)
    * MLP: ~35–40%
    * adaLN + residual: ~5%
    * Cache range [8,20) = 12 blocks, interval=2니까 절반의 step에서 12 blocks 를 건드립니다.

* Attention-only skip (attention은 cache, MLP는 매번 새로 계산):
    * cache된 step에서 block당 ~55% 절약 → 전체 1024px 기준 이론상 ~15–18% speedup.
    * 실측 기대: 1.15–1.18× (no-cache 20 step 대비).
    * 현재 DeepCache 1.22× 대비 조금 낮지만 FID 손해가 훨씬 작을 것.

* MLP-only skip:
    * ~10–12% speedup 기대.
    * 실측 기대: 1.10–1.13×.
    * 더 보수적이지만 quality 보존에 유리.

* 중요한 포인트: partial skip의 진짜 가치는 "speedup 극대화"가 아니라 "같은 speedup에서 품질 보존" 입니다. 

* Full block skip (현재 1.22×, FID 129.14) vs attention-only skip (예상 1.17×, FID 124–126 기대) — 

* speedup은 약간 손해지만 FID는 명확히 개선될 가능성이 높아요. 

* 지금 table을 보면 FID 3–4만 개선돼도 no-cache 121.32에 근접하니 의미 큽니다.

---



## 실험 설정

| 파라미터 | 값 |
|----------|-----|
| cache_mode | `partial_attn` / `partial_mlp` / `partial_attn_mlp` |
| cache_start / end | 8 / 20 |
| deepcache_interval | 2 |
| num_steps | 20 |
| num_samples | 100 |
| quant_method | SVDQUANT |

## 결과 (n=100, steps=20)

| 방법 | FID↓ | CLIP↑ | tpi(s)↓ | speedup↑ | 비고 |
|------|------|-------|---------|---------|------|
| SVDQUANT (no cache) | 121.32 | 34.840 | 2.85 | 1.00× | 기준 |
| DeepCache c8-20 (full) | 129.14 | 34.742 | 2.33 | 1.22× | 기준 |
| nl_gelu drift | 124.94 | 34.823 | 2.53 | 1.13× | 기준 |
| `partial_attn` | 135.60 | 34.623 | 2.71 | 1.05× | full skip보다 나쁨 |
| `partial_mlp` | 361.06 | 27.302 | 2.82 | 1.01× | **붕괴** |
| `partial_attn_mlp` | 128.72 | 34.639 | 2.60 | 1.10× | full skip 대비 미미한 개선 |

## 분석

### partial_mlp 붕괴 (FID=361)

ff의 입력 `norm2(hidden_states)` 는 self-attn과 cross-attn 이후의 누적된 hidden_states를 기반으로 하며, 이는 매 diffusion step마다 크게 변한다. Pre-gate ff output을 그대로 재사용하면 현재 step의 hidden_states 분포와 전혀 맞지 않아 결과가 발산함.

반면 attn1의 입력 `norm1(hidden_states)` 는 block 입력에 직접 가깝고, 인접 step 간 hidden_states 변화가 상대적으로 작아 partial_attn이 생존 가능했던 것으로 추정.

### partial_attn (FID=135.6) — 예상보다 나쁨

기존 DeepCache(129.14)보다 FID가 오히려 나쁨. 원인 분석:
- Self-attention output은 단순히 "작은 변화"가 아닐 수 있음. Block [8,20) 에서 attn1이 전체 diffusion trajectory 정보를 크게 갱신하는 역할을 함.
- Full block skip에서는 stale residual 전체가 하나의 단위로 재사용되어 내부적으로 일관성이 있지만, partial skip은 attn1은 stale + attn2와 ff는 fresh → 서로 다른 시점의 정보가 혼합됨.

### partial_attn_mlp (FID=128.72) — 미미한 개선

attn2 재계산 효과는 FID 기준 0.4pt 개선에 그침. Cross-attention은 block FLOPs의 ~3%이므로 speedup 이득도 거의 없음 (1.10× vs full skip 1.22×).

## 결론

**Pre-gate submodule caching 전략은 효과적이지 않음**. 특히:
- MLP(ff) pre-gate caching은 입력 분포 변화로 인해 완전 붕괴
- Self-attn(attn1) caching도 full block skip보다 FID가 나쁨
- Full block skip + nl_gelu drift corrector (FID=124.94)가 여전히 최고

본 실험이 시사하는 바: DeepCache의 강점은 "전체 block residual을 하나의 단위로 재사용"하는 내부 일관성에 있으며, 이를 submodule 수준으로 쪼개면 일관성이 깨져 오히려 품질이 나빠짐.

---
