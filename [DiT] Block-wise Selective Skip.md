
### Block-wise Sensitivity 기반 Selective Skip

* 현재 [8,20) 전체 12 blocks를 무차별 skip. Sensitivity 측정 후 하위 8개만 skip한다고 하면:
    * Skip 비율이 12 → 8로 줄어서 speedup은 1.22× → ~1.15–1.17× 로 감소
    * FID는 꽤 크게 개선 기대 (치명적인 block 4개를 살렸으니까). FID 125 근처 예상.

* 반대로 상위 sensitivity blocks까지 포함해서 더 넓게 (16 blocks) skip하면:
    * Speedup 1.22× → ~1.30× 상승 가능
    * 단 FID는 오히려 악화 — 현재 c2-26 (24 blocks) 이 147.42로 망가진 걸 보면 trade-off 분명

* Sweet spot은 거의 확실히 "더 좁게 + quality 개선" 방향입니다.
* Speedup 절대값으로는 큰 gain 없어요.


---
