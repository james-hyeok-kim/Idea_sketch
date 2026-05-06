Key idea, Toekn wise, quantization, caching 3가지를 최적화해서 가속해보자

### Saliency-Driven Resource Budgeting (Heuristic-Search)

* 토큰의 중요도(Saliency)를 중심으로 나머지 두 변수를 할당하는 전략입니다.
* 단계 1: Token Saliency 측정 ( $T$ ) Attention Map의 가중치( $\text{Attn Score}$ )나 Gradient의 크기를 사용하여 각 토큰의 중요도를 계산합니다.
* 단계 2: Joint Mapping Function 설계중요도 $S_i$에 따라 ${Q, C}$를 결정하는 함수 $f(S_i)$를 정의합니다.
    * $S_i$가 상위 10% (중요): $Q = \text{FP8}$, $C = \text{No-Cache}$
    * $S_i$가 하위 50% (배경): $Q = \text{NVFP4}$, $C = \text{Heavy-Cache}$ (예: 4스텝 유지)
* 실험 포인트: 이 매핑 함수 $f$의 파라미터를 Evolutionary Algorithm (유전 알고리즘)이나 Bayesian Optimization으로 최적화하세요.
    * "어떤 중요도 임계값에서 변수를 전환하는 것이 최적인가?"를 찾는 과정이 논문의 핵심 Contribution이 됩니다.

### 통계적 토큰 할당 함수 정의

각 토큰 $i$의 중요도 $S_i$에 따른 Quantization ($Q$)과 Caching ($C$)의 통합 할당 함수 $f(S_i)$는 다음과 같이 정의할 수 있습니다. 

$$
f(S_i) = 
\begin{cases} 
\{Q = \text{FP8}, \text{ } C = \text{Compute}\} & \text{if } S_i \geq \mu + \alpha\sigma \quad \text{(High Saliency)} \\
\{Q = \text{INT8}, \text{ } C = \text{Compute}\} & \text{if } \mu - \beta\sigma \leq S_i < \mu + \alpha\sigma \quad \text{(Mid Saliency)} \\
\{Q = \text{NVFP4}, \text{ } C = \text{Cache}\} & \text{if } S_i < \mu - \beta\sigma \quad \text{(Low Saliency)}
\end{cases}
$$


### 💡 [핵심 아이디어] 토큰 중요도( $S_i$ )는 어떻게 측정할 것인가?

* 함수 $f(S_i)$를 잘 정의하려면, 입력으로 들어가는 $S_i$ 자체가 정확해야 합니다. 추가 연산 없이 DiT에서 $S_i$를 공짜로 얻는 방법은 두 가지가 있습니다.

#### Cross-Attention Map 활용 (Conditioning Saliency):

* DiT는 타임스텝($t$)과 텍스트 프롬프트(Text/Label)를 Cross-attention으로 주입합니다.
* 이때 Text Token에 높게 반응하는 Image Token의 Attention 가중치 합을 $S_i$로 사용하세요.
* (즉, 조건에 반응하는 핵심 물체일수록 중요도가 높음)

#### Temporal Difference (시간적 변화량)

* 이전 타임스텝( $t+1$ )과 현재 타임스텝( $t$ ) 사이의 토큰 특징(Feature) 벡터의 차이( $||x_t - x_{t+1}||_2$ )를 $S_i$로 둡니다.
* 이전 스텝과 차이가 없으면(배경) 캐싱과 양자화를 세게 걸고, 차이가 크면 정밀하게 연산하는 매우 직관적인 방식입니다.

#### 추천하는 실험 진행 순서:

* 처음부터 복잡한 클러스터링을 구현하기보다는, 2번(Temporal Difference)으로 $S_i$를 구한 뒤,
* 1번(통계적 Z-score 분할)을 적용하여 $\alpha, \beta$를 조절해 보시는 것을 권장합니다.


### 📊 DiT 최적화를 위한 중요도(Saliency) 지표 전략적 추천

| 연구 단계 (Usage) | 권장 지표 (Metric) | 상세 이유 (Reasoning) |
| :--- | :--- | :--- |
| **Offline Analysis** (연구/분석) | **Hessian Trace** | 모델의 수렴 안정성과 양자화 민감도를 분석하는 가장 정확한 이론적 근거 제공. 주로 논문의 Figure 1(Motivation) 작성을 위한 정적 프로파일링에 사용. |
| **Online Routing** (실시간 추론) | **Attention Score + Temporal Diff** | 추가 연산 비용이 거의 없어 추론 속도에 영향을 주지 않음. 매 타임스텝(t)마다 변하는 토큰의 중요도를 실시간으로 반영하여 {Q, C} 조합을 동적으로 결정하기에 최적. |
| **Error Refinement** (오차 보정) | **Fisher Information** | Hessian의 효율적인 근사치로, 양자화 및 캐싱으로 발생한 오차를 Cache-LoRA가 어느 토큰에서 우선적으로 복구해야 할지(Prioritization) 판단하는 기준으로 적합. |
| **System Evaluation** (시스템 평가) | **Jacobian Frobenius Norm** | 입력 변화가 출력층까지 미치는 영향도를 측정. 특정 블록의 연산 효율화가 전체 이미지 생성 품질(FID)에 미치는 파급 효과를 수치화할 때 활용. |

### Gumbel Softmax

* Gumbel-Softmax는 이산적인 선택지($k$개) 중 하나를 샘플링하는 과정을 미분 가능하게 변환하는 Reparameterization Trick입니다.
* Training based search
    * "선택지를 골라야 하지만, 처음부터 끝까지 하나의 파이프라인으로 학습(End-to-End Learning)시키고 싶을 때" 활용
    * Mixed-Precision Quantization (혼합 정밀도 양자화)
        * "어떤 레이어에 어떤 비트(Bit)를 할당할 것인가?"를 결정할 때 쓰입니다.
    * Conditional Computation (조건부 연산) & Dynamic Routing
        * "이 연산을 할 것인가, 말 것인가(Skip/Cache)?" 혹은 "어느 경로로 보낼 것인가?"를 결정할 때 쓰입니다.
    * Differentiable Neural Architecture Search (DNAS)
        * "최적의 신경망 구조(Architecture) 자체를 인공지능이 찾게 만들자"는 연구 분야에서 쓰입니다. 

1. 일반 Softmax (Static Distribution)

* 단순히 입력 Logit ( $h_i$ )을 0과 1 사이의 확률값으로 변환합니다.
* 결과는 항상 고정된(Deterministic) 분포입니다.

$$P_i = \frac{\exp(h_i)}{\sum_{j=1}^k \exp(h_j)}$$

2.  Gumbel-Softmax (Stochastic Approximation)

* 입력 Logit에 Gumbel Noise ( $G_i$ )를 더해 무작위성을 부여하고,
* Temperature ( $\tau$ )로 결과의 분포를 조절합니다.

$$y_i = \frac{\exp((h_i + G_i) / \tau)}{\sum_{j=1}^k \exp((h_j + G_j) / \tau)}$$

$$G_i = -\log(-\log(U_i)), \quad U_i \sim \text{Uniform}(0, 1)$$
