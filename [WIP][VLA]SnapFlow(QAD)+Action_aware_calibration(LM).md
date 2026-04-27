# VLA Quant Idea

## Key Idea

* VLA-DiT: SnapFlow + Qant / Quant + SnapFlow가 아니라, Quantization aware Snapflow를 하고 Int4 Quant
* VLA-LM: Action aware calibration으로  Int4 Quant


## Flow matching step 축소 (최고 ROI): 

SnapFlow는 10-step denoising을 1-step으로 self-distillation해서 π0.5에서 9.6배 denoising 가속, 

E2E는 274ms → 83ms를 달성하면서 성공률은 오히려 약간 상승 (97.75% → 98.75%) arXiv.

Consistency model/shortcut model 계열을 π0.5에 붙이는 게 가장 임팩트 큼.

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

## LM
LM 은 
