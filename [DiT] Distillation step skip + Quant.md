### Step Distillation (DMD2 / HyperSD / SANA-Sprint)

이게 압도적으로 speedup이 큽니다. 나머지 방법들과 자릿수가 달라요.

* 현재 no-cache baseline:
    * 20 steps: 2.85s (1.00×)
    * 10 steps: 1.44s (2.0×)

* Step distillation 적용 시:
    * 8 steps: ~1.15s → 2.5× speedup
    * 4 steps: ~0.60s → 4.75× speedup
    * 2 steps: ~0.32s → 8.9× speedup
    * 1 step: ~0.18s (VAE decode 포함) → ~15× speedup

* HyperSD나 DMD2가 PixArt-Sigma에 직접 공개되어 있지는 않지만, FLUX/SD3에서 4-step에서 quality가 거의 유지되는 결과를 보여줬어요.
* PixArt-Sigma에서도 비슷할 거라 예상됩니다.
* Distillation training 자체는 비용이 들지만 (수 GPU-day), inference speedup은 확정적.
* SVDQuant 조합 시 주의점:
    * Distilled few-step model은 per-step signal이 더 강하고 trajectory가 짧아서 quant error에 더 민감할 수 있어요.
    * 4-step + 4-bit 조합이 4-step + 8-bit보다 FID가 훨씬 나쁠 가능성 있음. 이게 바로 논문의 failure mode 분석 포인트가 됩니다.

---
