Key idea, Toekn wise, quantization, caching 3가지를 최적화해서 가속해보자

### Saliency-Driven Resource Budgeting (Heuristic-Search)

* 토큰의 중요도(Saliency)를 중심으로 나머지 두 변수를 할당하는 전략입니다.
* 단계 1: Token Saliency 측정 ( $T$ ) Attention Map의 가중치( $\text{Attn Score}$ )나 Gradient의 크기를 사용하여 각 토큰의 중요도를 계산합니다.
* 단계 2: Joint Mapping Function 설계중요도 $S_i$에 따라 ${Q, C}$를 결정하는 함수 $f(S_i)$를 정의합니다.
    * $S_i$가 상위 10% (중요): $Q = \text{FP8}$, $C = \text{No-Cache}$
    * $S_i$가 하위 50% (배경): $Q = \text{NVFP4}$, $C = \text{Heavy-Cache}$ (예: 4스텝 유지)
* 실험 포인트: 이 매핑 함수 $f$의 파라미터를 Evolutionary Algorithm (유전 알고리즘)이나 Bayesian Optimization으로 최적화하세요.
    * "어떤 중요도 임계값에서 변수를 전환하는 것이 최적인가?"를 찾는 과정이 논문의 핵심 Contribution이 됩니다.
