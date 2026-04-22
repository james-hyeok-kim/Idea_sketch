# VLA Quant Idea

## Hadamard rotation (QuaRot/SpinQuant): 

HQ-DiT가 Hadamard transform을 DiT에 적용해 W4A4까지 가능케 했음 arXiv. LLM의 QuaRot와 동일한 수학이라 같이 적용 가능.

## Flow matching step 축소 (최고 ROI): 

SnapFlow는 10-step denoising을 1-step으로 self-distillation해서 π0.5에서 9.6배 denoising 가속, 

E2E는 274ms → 83ms를 달성하면서 성공률은 오히려 약간 상승 (97.75% → 98.75%) arXiv.

Consistency model/shortcut model 계열을 π0.5에 붙이는 게 가장 임팩트 큼.



전체 로드맵
```
Stage 0: Baseline 재현 및 벤치마크 확정
Stage 1: SnapFlow distillation (1-NFE)
Stage 2: QuaRot - LLM만 적용
Stage 3: QuaRot - DiT까지 확장
Stage 4: 경계 보정 (OHB) 및 AdaLN 처리
Stage 5: W4A4 공격적 quant
Stage 6: End-to-end 통합 최적화
```
각 stage는 "이전 대비 얼마나 손해 봤는가" 를 추적 가능하게 설계했어요.

---

### Snapflow (10 → 1): Progressive Self-Distillation

* Flow-matching 모델은 노이즈 $x_1$에서 데이터 $x_0$로 가는 궤적을 미분 방정식(ODE)으로 정의

$$\frac{dx_t}{dt} = v_\theta(x_t, t)$$

* $v_\theta$는 모델이 예측한 **속도장(velocity field)**입니다. 일반적인 추론에서는 이 $v$를 따라 작은 보폭($\Delta t$)으로 여러 번 이동해야 정답에 도달합니다.

#### Step 1: Teacher의 시뮬레이션 (2-step)

모델(현재의 자기 자신)을 사용하여 현재 시점 $t$에서 두 번의 짧은 Euler Step을 밟아봅니다.

첫 번째 점프: 

$$\hat{x}_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t)$$

두 번째 점프: 

$$\hat{x}_{t-2\Delta t} = \hat{x}_{t-\Delta t} - \Delta t \cdot v_\theta(\hat{x}_{t-\Delta t}, t-\Delta t)$$

이 결과물인 $\hat{x}_{t-2\Delta t}$가 **"정답(Target)"**이 됩니다.

#### Step 2: Student의 학습 (1-step Shortcut)

이제 Student 모델(학습 대상)에게 미션을 줍니다.

"너는 $x_t$에서 한 번만 계산해서, 아까 Teacher가 두 번 걸려서 간 $\hat{x}_{t-2\Delta t}$ 위치에 바로 도착해!"

즉, 아래의 손실 함수를 최소화합니다.

$$\mathcal{L} = \| v_\theta^{student}(x_t, t) - v_{target} \|^2$$

```
16단계 → 8단계로 압축 (2개씩 묶기)
8단계 → 4단계로 압축
... 최종적으로 1단계(1-NFE) 달성
```


## Zero-initialized Target-time Embedding

만약 이 정보를 주기 위해 새로운 임베딩 레이어를 추가하고 랜덤하게 초기화한다면:

학습 초기 단계에서 모델의 출력이 완전히 망가집니다.

기존에 모델이 가지고 있던 정교한 로봇 제어 능력(Prior)을 다시 처음부터 학습해야 하는 비효율이 발생합니다.


## OHB

**OHB(Outlier-aware Hybrid Balancing)**는 양자화 오차로 인해 발생하는 출력 분포의 '드리프트(Drift)'를 교정합니다.


### 2. $\beta$: 하이브리드 밸런싱 (Outlier-aware Hybrid Balancing, OHB)

$\beta$는 활성화 값($X$)에 쏠린 양자화 난이도를 가중치( $W$ )로 강제로 옮겨오는 Smoothing 강도를 조절합니다. 

SmoothQuant 계열 연구에서 핵심이 되는 수식입니다.

수식: Scaling Factor 계산, 각 채널 $j$에 대해 스케일링 팩터 $s_j$를 다음과 같이 계산합니다.

$$s_j = \frac{(\max |X_j|)^\beta}{(\max |W_j|)^{1-\beta}}$$

적용 (In-place Migration)

계산된 $s$를 사용하여 가중치와 활성화 값을 변환합니다.

$$X' = X \cdot \text{diag}(s)^{-1}$$

$$W' = \text{diag}(s) \cdot W$$

$\beta = 0.5$ (Balanced): 가중치와 활성화 값의 아웃라이어 압력을 공평하게 나눕니다.

$\beta > 0.5$: 활성화 값의 아웃라이어가 너무 심할 때, 가중치를 더 키우더라도 활성화 값의 범위를 대폭 줄여서 양자화 안정성을 확보합니다.

VLA에서의 의미: 모델 추론 시 특정 Attention Head에서 발생하는 수치적 불안정성을 $\beta$ 조절을 통해 해결합니다.

## ATM

1. $\alpha$: 활성화 임계치 매칭 (Activation Threshold Matching, ATM)

$\alpha$는 활성화 값의 분포 중 어디까지를 '유효한 범위'로 볼 것인지 결정하는 파라미터입니다. 주로 Symmetric Quantization 환경에서 스케일($\Delta$)을 조정할 때 사용됩니다.

수식

$$\Delta = \alpha \cdot \frac{\max(|X|)}{2^{b-1} - 1}$$

$$X_{quant} = \text{clamp}\left( \left\lfloor \frac{X}{\Delta} \right\rceil, -Q_{max}, Q_{max} \right)$$


## 3. 종합 적용: QuantVLA의 목적 함수

최근 연구에서는 이 두 파라미터를 고정하지 않고, 원본 FP16 모델과 양자화 모델 간의 출력 오차(MSE)를 최소화하는 방향으로 최적화합니다.

$$\min_{\alpha, \beta} \| \text{Attn}(X, W) - \text{Attn}_{quant}(X', W', \alpha, \beta) \|^2$$


<img width="452" height="269" alt="image" src="https://github.com/user-attachments/assets/f1a53acc-aebb-4adb-9a12-3ec4be90e3b1" />


<img width="662" height="248" alt="스크린샷 2026-04-22 오후 3 21 07" src="https://github.com/user-attachments/assets/39d23ee2-8182-4b5e-b5e9-9b721b6b8dc2" />

<img width="805" height="373" alt="스크린샷 2026-04-22 오후 3 20 34" src="https://github.com/user-attachments/assets/5968d918-156e-4748-99f8-34d5dc2b2af6" />


