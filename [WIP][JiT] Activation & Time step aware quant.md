## JiT

JiT의 

JiT-H/16 모델의 활성화(activation)는 heavy-tail 분포(kurtosis mean=599.8, abs_max≈712)를 가지며 outlier가 심하다.
이를 W4A4로 양자화하면서 quality 손실을 줄이는 기법들을 비교한다.

**목표**: Hadamard pre-rotation, timestep-bucketed smoothing, Hessian-weighted SVD를 조합한 Phase 2 기법이 기존 PTQ SOTA 대비 어느 위치에 있는지 정량화.

**평가 설정**: ImageNet 256×256, stratified 100 samples (10 classes × 10 images), 동일 (label, noise seed) 페어 전 phase 공통. FID + IS 측정. 실제 INT4/FP4 CUDA 커널이 아닌 fake quantization.


옵션 A로 가시되, 다음 구조를 추천합니다.

1) 분석 챕터: 이미 수행한 weight/activation/timestep 분석을 pixel-space vs latent-space 비교로 재구성. 이것만으로도 강한 motivation이 됨

2) Failure mode 분석: SVDQuant, TR-DQ 등을 JiT/PixelDiT에 그냥 적용했을 때 어디서 깨지는지 정량 분석. 이게 두 번째 contribution

### Idea 1 (Rotation + timestep smoothing(bucket) + Hessian SVD)



3) 솔루션: 세 기법 조합.
    1) 단, 각 기법이 어느 failure mode를 푸는지 명확히 매핑(rotation→heavy-tail activation, timestep smoothing→PixelDiT의 후반 step outlier 증가, Hessian SVD→outlier 레이어 보존).
    2) 매핑이 명확해야 "조합"이 아니라 "principled design"으로 읽힘

4) 검증:
    1) JiT + PixelDiT(patch/pixel 둘 다) 두 모델에서 baseline 5-6개 대비 정량화.
    2) PixArt도 한 번 돌려서 "latent-space에서는 간단한 베이스라인으로도 충분하지만 pixel-space에서는 우리 방법이 필요하다"를 보여줌

### Idea 2 (Activation aware scale quantization)

* NVFP4 or MXFP4 로 Activation,Weight Quant하고 Outlier(Kurtosis) 높은것만 Activation scale을 더 정교한 것으로 반영하고 싶어, FP16 / BF16/FP32 처럼
