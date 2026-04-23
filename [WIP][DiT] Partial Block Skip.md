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

