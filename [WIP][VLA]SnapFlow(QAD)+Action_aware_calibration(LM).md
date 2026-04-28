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


## QAD

1. 문제점: 오일러 방법(Euler Method)의 한계

* 보통 $NFE=1$(Network Function Evaluation, 신경망 호출 횟수)이라는 건, 현재 위치에서의 기울기(Velocity)만 보고 다음 위치를 때려 맞추는 오일러 방법을 씁니다.
* 비유: 당신이 커브길에서 운전을 하는데, 핸들을 처음 진입할 때의 각도로만 고정하고 100m를 가는 것과 같습니다.
* 당연히 길 밖으로 튕겨 나가겠죠?
* 수식으로 보면: $x_{t+1} = x_t + v(x_t, t)$이 방식은 $t$와 $t+1$ 사이의 곡률(Curvature)을 전혀 반영하지 못해 오차가 큽니다.

2. 해결책: 사다리꼴 공식 (Trapezoidal Rule)

* 사다리꼴 공식은 시작점의 기울기와 도착할 지점의 기울기를 평균 내서 사용합니다.
* 시작점 기울기( $v_1$ ): 현재 $x_t$에서의 기울기.
* 예측 도착점: $x_t$에서 $v_1$ 방향으로 일단 가본 지점 ( $x_{next}$ ).
* 도착점 기울기( $v_2$ ): 그 가본 지점( $x_{next}$ )에서의 기울기.
* 최종 타겟: $\frac{v_1 + v_2}{2}$ (이 두 기울기의 평균값)

## LM
LM 은 
