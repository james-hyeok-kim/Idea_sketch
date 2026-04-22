실전 구조 — 3 단계로 올리세요
구현 비용 대비 gain이 큰 순서대로 배열했어요. 각 단계마다 FID 다시 찍고 의미 있는 gain이 있으면 다음 단계로.

## Level 1: Feature Distillation (가장 쉽고 즉효 큼)

Target을 바꾸세요. "residual의 residual" 말고, teacher (full forward) 의 block output을 직접 맞추기:

```
python
#Teacher: full forward (no cache)
with torch.no_grad():
    h_teacher = transformer_blocks[cache_start:cache_end](h_in)

# Student: cache + corrector
h_cached = cached_features  # from previous step
h_student = corrector(h_in, h_cached, t)

loss = F.mse_loss(h_student, h_teacher)
```

이렇게만 바꿔도 signal이 훨씬 강해집니다. 지금 structure (FiLM + GELU + MLP) 그대로 쓰시되 target만 변경.

추가 중요 포인트 — 

Loss를 token-wise로 reweight하세요. 

매 batch에서 token별 teacher output의 variance를 재서, variance 큰 token에 더 큰 weight. 이게 "background dominance" 문제를 부분적으로 풀어줍니다:

```
python
token_var = h_teacher.var(dim=-1, keepdim=True)  # per-token variance
weight = token_var / token_var.mean()
loss = (weight * (h_student - h_teacher).pow(2)).mean()
```

## Level 2: Timestep-aware Corrector + Timestep-stratified Loss

지금 FiLM만 t_norm을 받고 나머지는 안 받는다고 하셨는데, 모든 서브모듈이 t_norm을 받도록 하세요.

간단히는 GELU 앞/뒤에 시간 의존 scale-shift 하나씩 더 넣거나, MLP를 timestep-conditioned MLP로 바꾸는 방법.

Loss 쪽은 timestep bucket별로 나눠서 평균:

```
python# T를 예를 들어 5개 bucket으로 나눔
bucket = (t * 5 / T).long()
per_bucket_loss = scatter_mean(loss_per_sample, bucket)
total_loss = per_bucket_loss.mean()  # bucket 평균의 평균
```

이렇게 하면 sample이 많이 뽑히는 timestep이 loss를 dominate하는 걸 막아줘요.

Early timestep error와 late timestep error가 평등하게 학습됨.

## Level 3: Trajectory Distillation (진짜 quality 복구)
이게 비용은 가장 크지만 효과도 가장 큽니다. Corrector만 학습된 상태에서 추가로:

```
python# K step rollout (K=4~8 정도, 전체는 말고)

x_t = noise
for step in sampled_steps:  # 전체 step 중 연속한 K개만
    x_t = corrector_forward(x_t, step)  # student trajectory
x_student = x_t

# 동일 starting point에서 teacher rollout
x_t = same_noise
for step in sampled_steps:
    x_t = full_forward(x_t, step)  # teacher trajectory
x_teacher = x_t.detach()

traj_loss = F.mse_loss(x_student, x_teacher)
total_loss = feature_loss + lambda_traj * traj_loss
```

핵심 팁:

전체 trajectory를 backprop하면 OOM 납니다. K=4~8 step의 window만 돌리세요

Window의 시작점을 random하게 sampling. 다양한 timestep 범위를 커버.

Teacher는 no_grad, student만 gradient.

이건 기존 feature loss 대체가 아니라 추가. Feature loss는 dense supervision, trajectory loss는 global correction. 둘 다 필요.

$λ_traj\lambda_{traj}$

$λ_traj$는 처음에 작게 (0.1 정도) 시작해서 올려가세요.


### Level 4 (Optional): Attention Map Distillation

지난번에 "softmax가 비선형이라서 attention distortion이 복구 안 된다"는 얘기를 했는데, 이걸 직접 공격하려면 teacher의 attention weight를 KL divergence로 맞추는 loss를 추가하세요:
pythonattn_loss = F.kl_div(
    student_attn.log(),
    teacher_attn,
    reduction='batchmean'
)

Rank를 올리는 것보다 이게 더 효과 클 수 있어요. Attention 구조 자체를 직접 supervision하니까.

예상 결과 & 권장 순서

경험적으로 이런 distillation 체인에서는:


Level 1만 해도 FID 3-5 정도는 떨어집니다 (지금 20-step 126.56 → 122 정도 기대 가능)

Level 2 추가로 timestep이질성이 큰 regime (15 step 같은)에서 특히 효과 — 가장 큰 gain이 15-step 쪽에서 나올 가능성 높음

Level 3이 FID의 "floor"를 내리는 역할 — 지금까지 안 잡히던 drift를 잡아줌

제 추천 순서:

먼저 Level 1만 구현하고 rank sweep 다시 돌리세요 (4/8/16). Target 바꾼 상태에서의 saturation point는 지금과 완전히 다를 거예요.

Rank가 여전히 빨리 saturate하면 Level 2로 넘어가기.

15-step quality가 여전히 부족하면 Level 3.
