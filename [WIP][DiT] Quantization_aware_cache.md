## DiT에서 Quant+Cache 문제점을 분석후 해결방안 찾기

## Quant + Cache에서 Error 가장 많이 나오는 곳은?

### 이미 존재하는 핵심 선행연구

* Q&C (Ding et al., 2025, arXiv:2503.02508) — "When Quantization Meets Cache in Efficient Image Generation"가 정확히 이 주제를 다룹니다. 핵심 발견 두 가지:
    * Cache 연산이 calibration dataset의 sample efficacy를 무너뜨린다 (timestep 간 cosine similarity가 cache 적용 후 급격히 높아져 redundancy 발생)
    * Quantization+Cache 결합 시 exposure bias가 더 심하게 누적된다
    * 이를 해결하기 위해 TAP (Temporal-Aware Parallel Clustering) 으로 cache-aware calibration data selection을 하고, VC (Variance Compensation) 으로 분포 보정을 합니다.

* CacheQuant — quantization과 caching이 "not entirely orthogonal"하다는 관찰에서 출발해 joint optimization을 수행한 training-free 프레임워크.
* QuantCache (2025) — hierarchical latent caching + adaptive importance-guided quantization + pruning 3개를 동시 최적화 (Open-Sora에서 6.72× 가속).
* Increment-Calibrated Caching (2025) — low-rank 가중치 근사를 calibration parameter로 두고 cached feature를 보정. 컨셉적으로 quantization-like 압축 + cache-aware calibration의 변형.

* 그래서 novelty가 가능한 지점
    * 기존 논문들은 대부분 DiT-XL/2, PixArt-α 위주이고 W8A8 / W4A8 PTQ + step-level cache (FORA, Δ-DiT 류) 조합에 집중되어 있습니다. 아직 비어있는 영역:

## DiT Quantization + Cache 4편 비교

| 항목 | Q&C (Ding et al., 2025) | CacheQuant (Liu et al., CVPR 2025) | QuantCache (Wu et al., 2025) | ICC (Chen et al., CVPR 2025) |
|---|---|---|---|---|
| **타겟 모델** | DiT-XL/2, LDM | DDPM, LDM, SD, DiT-XL/2 | Open-Sora 1.2 (Video DiT) | DiT-XL/2, PixArt-α |
| **Weight bit-width** | W8 (256², LDM) / W4 (512²) | W8 (main), W4 (reconstruction 필요) | {W4, W8, FP16} mixed | FP16 (양자화 없음) |
| **Activation bit-width** | A8 | A8 | {A4, A6, A8} mixed | FP16 (양자화 없음) |
| **주요 보고 config** | W8A8, W4A8 | W8A8 | W8A8, W4A6 | FP16 |
| **Weight granularity** | Channel-wise | Channel-wise | Per-channel (offline) | — |
| **Activation granularity** | Tensor-wise (per-tensor) | Layer-wise (per-layer) | Per-layer dynamic (online) | — |
| **Quantizer 종류** | Uniform (PTQ4DiT 베이스) | Uniform (EDA-DM + temporal quantizer) | Uniform min-max + scale/rotation balancing | N/A |
| **Mixed precision** | ✗ | ✗ | ✓ (layer/timestep별 동적 배분) | ✗ |
| **Block/Group quant** | ✗ | ✗ | ✗ | ✗ |
| **Low-rank / LoRA** | ✗ | ✗ | ✗ | ✓ (SVD 기반) |
| **Low-rank rank** | — | — | — | DiT-XL/2: r ∈ {128, 192, 256} / PixArt-α: r = 64 |
| **Low-rank precision** | — | — | — | FP16 |
| **Cache 방식** | Learn-to-Cache | DeepCache (UNet) / Δ-DiT (DiT) | HLC (adaptive feature divergence 기반) | FORA pattern (period p=2 or 3) |
| **Calibration data** | TAP clustering으로 800 samples | Reconstruction 기반 | Small calibration set (offline) | 256 images (random sampling) |
| **추가 보정 모듈** | VC (variance compensation) | DEC (channel-wise affine 보정 a, b) | SRAP (layer pruning) | Channel-aware SVD (CA-SVD / CD-SVD) |
| **Training-free** | ✓ | ✓ (W8A8) / ✗ (W4A8) | ✓ | ✓ |
| **주요 성과 (speedup)** | 12.7× (DiT, ImageNet 256²) | 5.18× (SD, MS-COCO) | 6.72× (Open-Sora, A800) | ~2× vs DDIM 35-step |
| **Quantization + Cache 결합** | ✓ (joint) | ✓ (joint, DPS schedule) | ✓ (joint + pruning) | ✗ (cache only, low-rank 보정) |
