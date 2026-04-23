### 10 Steps에서 Quant 낮추기 (Step-aware Mixed Precision)

* SVDQuant 4-bit이 FP16 대비 이미 ~2–2.5× 속도 이득을 내는 상태고,
* 여기서 3-bit이나 2-bit로 내려가도 kernel 효율상 추가 이득이 10–20% 수준입니다
    * (메모리 bandwidth는 줄지만 compute는 크게 안 줄어듦).

* Current: SVDQuant 4-bit, 10 steps, 1.44s
* Mixed 3/4-bit (민감 step만 4-bit, 나머지 3-bit):
    * ~1.25–1.30s 기대. 1.10–1.15× 추가 gain
    * 2-bit까지 내리면 kernel 효율 떨어져서 오히려 느려질 가능성 있음.
    * Inference speed 관점에서는 W4A4 이하는 실익이 적음.

* 이 방향의 진짜 가치는 speedup이 아니라 메모리 footprint입니다.
* Edge device 배포 / batch size 늘리기 용도.
* NeurIPS 논문 angle로는 "quant의 timestep 민감도 분석" 자체가 기여.
