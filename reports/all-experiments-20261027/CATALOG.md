# 전체 모델별 최종 기록

분할·학습 조건이 다른 행을 직접 순위 비교하지 마세요. `unknown`은 근거가 부족한 분할입니다. 마지막 기록이며 완료를 보장하지 않습니다.

| 실행 | 모델 | 평가 | epoch | 정확도 | Macro-F1 | 원본 |
|---|---|---|---:|---:|---:|---|
| main/Semi-supervised-learning-main/dual/result/svhn_smoke_minor | minor | validation | 1 | 9.88 | 2.86 | [파일](raw/main/0c3073960325/per_class_eval.csv) |
| main/Semi-supervised-learning-main/dual/result/svhn_smoke_minor_cuda | minor | validation | 1 | 9.88 | 2.86 | [파일](raw/main/268ad25bb6a0/per_class_eval.csv) |
| main/Semi-supervised-learning-main/dual/result/svhn_smoke_optimized_loader | minor | validation | 1 | 10.08 | 3.23 | [파일](raw/main/1a413e396632/per_class_eval.csv) |
| main/Semi-supervised-learning-main/dual/result/svhn_smoke_optimized_loader2 | minor | validation | 1 | 10.00 | 3.02 | [파일](raw/main/f2bd1cbc36b4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/dual/results/speed_probe | major | unknown | 1 | 9.32 | 2.12 | [파일](raw/main/c1d67996e927/per_class_eval.csv) |
| main/Semi-supervised-learning-main/dual/results/speed_probe | minor | unknown | 1 | 9.99 | 1.82 | [파일](raw/main/c1d67996e927/per_class_eval.csv) |
| main/Semi-supervised-learning-main/dual/results/speed_probe | ensemble | unknown | 1 | 10.00 | 1.82 | [파일](raw/main/c1d67996e927/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/baseline_fixmatch | fixmatch | unknown | 50 | 76.03 | 75.70 | [파일](raw/main/7d350e8c9537/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_none | major | unknown | 50 | 86.28 | 86.43 | [파일](raw/main/b6cd7e8026f9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_none | minor | unknown | 50 | 86.03 | 86.19 | [파일](raw/main/b6cd7e8026f9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_none | ensemble | unknown | 50 | 86.43 | 86.59 | [파일](raw/main/b6cd7e8026f9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_sampler | major | unknown | 50 | 85.65 | 85.81 | [파일](raw/main/ea7d65010ac7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_sampler | minor | unknown | 50 | 86.18 | 86.35 | [파일](raw/main/ea7d65010ac7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_sampler | ensemble | unknown | 50 | 86.35 | 86.51 | [파일](raw/main/ea7d65010ac7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_sampler_weighted_ce | major | unknown | 50 | 85.67 | 85.82 | [파일](raw/main/d8b02576318c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_sampler_weighted_ce | minor | unknown | 50 | 88.31 | 88.41 | [파일](raw/main/d8b02576318c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_sampler_weighted_ce | ensemble | unknown | 50 | 88.29 | 88.39 | [파일](raw/main/d8b02576318c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_weighted_ce | major | unknown | 50 | 85.08 | 85.24 | [파일](raw/main/42b3d3701a3c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_weighted_ce | minor | unknown | 50 | 85.68 | 85.84 | [파일](raw/main/42b3d3701a3c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/dual_weighted_ce | ensemble | unknown | 50 | 85.85 | 86.00 | [파일](raw/main/42b3d3701a3c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/baseline_fixmatch | fixmatch | unknown | 50 | 76.05 | 75.81 | [파일](raw/main/f19904b17ac6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_none | major | unknown | 50 | 85.93 | 86.08 | [파일](raw/main/8501b46cf15a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_none | minor | unknown | 50 | 85.03 | 85.23 | [파일](raw/main/8501b46cf15a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_none | ensemble | unknown | 50 | 85.18 | 85.36 | [파일](raw/main/8501b46cf15a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_sampler | major | unknown | 50 | 85.81 | 85.99 | [파일](raw/main/072d28acb1c1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_sampler | minor | unknown | 50 | 86.49 | 86.63 | [파일](raw/main/072d28acb1c1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_sampler | ensemble | unknown | 50 | 86.72 | 86.85 | [파일](raw/main/072d28acb1c1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_sampler_weighted_ce | major | unknown | 50 | 85.75 | 85.89 | [파일](raw/main/e5b55223d1c9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_sampler_weighted_ce | minor | unknown | 50 | 88.34 | 88.44 | [파일](raw/main/e5b55223d1c9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_sampler_weighted_ce | ensemble | unknown | 50 | 88.20 | 88.30 | [파일](raw/main/e5b55223d1c9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_weighted_ce | major | unknown | 50 | 84.80 | 84.96 | [파일](raw/main/79defc2dab5a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_weighted_ce | minor | unknown | 50 | 84.51 | 84.69 | [파일](raw/main/79defc2dab5a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/dual_weighted_ce | ensemble | unknown | 50 | 84.82 | 85.00 | [파일](raw/main/79defc2dab5a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/baseline_fixmatch | fixmatch | unknown | 50 | 77.95 | 77.81 | [파일](raw/main/c4009c65a0b3/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_none | major | unknown | 50 | 86.03 | 86.13 | [파일](raw/main/47ebb8696d2d/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_none | minor | unknown | 50 | 85.60 | 85.71 | [파일](raw/main/47ebb8696d2d/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_none | ensemble | unknown | 50 | 85.94 | 86.04 | [파일](raw/main/47ebb8696d2d/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_sampler | major | unknown | 50 | 85.76 | 85.87 | [파일](raw/main/0a14342a13b0/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_sampler | minor | unknown | 50 | 86.52 | 86.62 | [파일](raw/main/0a14342a13b0/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_sampler | ensemble | unknown | 50 | 86.73 | 86.84 | [파일](raw/main/0a14342a13b0/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_sampler_weighted_ce | major | unknown | 50 | 85.91 | 86.03 | [파일](raw/main/cbd813055c19/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_sampler_weighted_ce | minor | unknown | 50 | 88.18 | 88.27 | [파일](raw/main/cbd813055c19/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_sampler_weighted_ce | ensemble | unknown | 50 | 88.23 | 88.32 | [파일](raw/main/cbd813055c19/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_weighted_ce | major | unknown | 50 | 84.86 | 84.99 | [파일](raw/main/6e9bd9658bc2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_weighted_ce | minor | unknown | 50 | 84.56 | 84.68 | [파일](raw/main/6e9bd9658bc2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/dual_weighted_ce | ensemble | unknown | 50 | 84.93 | 85.06 | [파일](raw/main/6e9bd9658bc2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/baseline_fixmatch | fixmatch | unknown | 50 | 81.48 | 81.60 | [파일](raw/main/95443ef2eea9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_none | major | unknown | 50 | 87.28 | 87.39 | [파일](raw/main/50f0f88fa5f4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_none | minor | unknown | 50 | 87.18 | 87.32 | [파일](raw/main/50f0f88fa5f4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_none | ensemble | unknown | 50 | 87.62 | 87.74 | [파일](raw/main/50f0f88fa5f4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_sampler | major | unknown | 50 | 87.14 | 87.25 | [파일](raw/main/cc4e7ee2f5bc/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_sampler | minor | unknown | 50 | 87.62 | 87.71 | [파일](raw/main/cc4e7ee2f5bc/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_sampler | ensemble | unknown | 50 | 87.93 | 88.03 | [파일](raw/main/cc4e7ee2f5bc/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_sampler_weighted_ce | major | unknown | 50 | 86.81 | 86.92 | [파일](raw/main/25a1f80cb7e3/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_sampler_weighted_ce | minor | unknown | 50 | 88.83 | 88.90 | [파일](raw/main/25a1f80cb7e3/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_sampler_weighted_ce | ensemble | unknown | 50 | 88.69 | 88.77 | [파일](raw/main/25a1f80cb7e3/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_weighted_ce | major | unknown | 50 | 86.78 | 86.88 | [파일](raw/main/01b15f23c162/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_weighted_ce | minor | unknown | 50 | 87.03 | 87.13 | [파일](raw/main/01b15f23c162/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/dual_weighted_ce | ensemble | unknown | 50 | 87.24 | 87.34 | [파일](raw/main/01b15f23c162/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/baseline_fixmatch | fixmatch | unknown | 50 | 82.71 | 82.85 | [파일](raw/main/e6b5664fe3c4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_none | major | unknown | 50 | 86.87 | 86.99 | [파일](raw/main/f983ffeee13b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_none | minor | unknown | 50 | 86.64 | 86.76 | [파일](raw/main/f983ffeee13b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_none | ensemble | unknown | 50 | 86.93 | 87.05 | [파일](raw/main/f983ffeee13b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_sampler | major | unknown | 50 | 87.19 | 87.31 | [파일](raw/main/e0719f4593b7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_sampler | minor | unknown | 50 | 88.06 | 88.15 | [파일](raw/main/e0719f4593b7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_sampler | ensemble | unknown | 50 | 88.43 | 88.52 | [파일](raw/main/e0719f4593b7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_sampler_weighted_ce | major | unknown | 50 | 87.22 | 87.35 | [파일](raw/main/5b908ee58ecd/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_sampler_weighted_ce | minor | unknown | 50 | 89.07 | 89.14 | [파일](raw/main/5b908ee58ecd/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_sampler_weighted_ce | ensemble | unknown | 50 | 89.07 | 89.14 | [파일](raw/main/5b908ee58ecd/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_weighted_ce | major | unknown | 50 | 86.23 | 86.36 | [파일](raw/main/f05fe2c613a7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_weighted_ce | minor | unknown | 50 | 86.49 | 86.62 | [파일](raw/main/f05fe2c613a7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/dual_weighted_ce | ensemble | unknown | 50 | 86.86 | 86.99 | [파일](raw/main/f05fe2c613a7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/baseline_fixmatch | fixmatch | unknown | 50 | 81.61 | 81.69 | [파일](raw/main/344433d30130/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_none | major | unknown | 50 | 87.62 | 87.71 | [파일](raw/main/fac60add211c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_none | minor | unknown | 50 | 86.92 | 87.03 | [파일](raw/main/fac60add211c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_none | ensemble | unknown | 50 | 87.28 | 87.39 | [파일](raw/main/fac60add211c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_sampler | major | unknown | 50 | 86.81 | 86.92 | [파일](raw/main/81fa166f747a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_sampler | minor | unknown | 50 | 87.24 | 87.33 | [파일](raw/main/81fa166f747a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_sampler | ensemble | unknown | 50 | 87.65 | 87.73 | [파일](raw/main/81fa166f747a/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_sampler_weighted_ce | major | unknown | 50 | 86.68 | 86.79 | [파일](raw/main/2c3efe5c61d9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_sampler_weighted_ce | minor | unknown | 50 | 88.75 | 88.81 | [파일](raw/main/2c3efe5c61d9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_sampler_weighted_ce | ensemble | unknown | 50 | 88.84 | 88.91 | [파일](raw/main/2c3efe5c61d9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_weighted_ce | major | unknown | 50 | 86.14 | 86.26 | [파일](raw/main/3662e5b6c41e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_weighted_ce | minor | unknown | 50 | 86.18 | 86.29 | [파일](raw/main/3662e5b6c41e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ablation_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/dual_weighted_ce | ensemble | unknown | 50 | 86.58 | 86.68 | [파일](raw/main/3662e5b6c41e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_2_3_ratio100_e50_s512/minor_only_alpha12_seed_2 | minor | validation | 3 | 40.22 | 37.46 | [파일](raw/main/4bb839025895/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_2 | major | validation | 50 | 86.90 | 86.96 | [파일](raw/main/4b058a88a613/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_2 | minor | validation | 50 | 89.12 | 89.16 | [파일](raw/main/4b058a88a613/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_2 | ensemble | validation | 50 | 89.06 | 89.09 | [파일](raw/main/4b058a88a613/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_3 | major | validation | 50 | 88.04 | 88.11 | [파일](raw/main/afb8efb73c57/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_3 | minor | validation | 50 | 89.72 | 89.75 | [파일](raw/main/afb8efb73c57/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_3 | ensemble | validation | 50 | 89.70 | 89.72 | [파일](raw/main/afb8efb73c57/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_4 | major | validation | 50 | 86.80 | 86.87 | [파일](raw/main/5e2a21d506f9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_4 | minor | validation | 50 | 88.70 | 88.75 | [파일](raw/main/5e2a21d506f9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_4 | ensemble | validation | 50 | 88.66 | 88.71 | [파일](raw/main/5e2a21d506f9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_5 | major | validation | 50 | 88.56 | 88.59 | [파일](raw/main/b06d3a3f9fb9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_5 | minor | validation | 50 | 90.40 | 90.41 | [파일](raw/main/b06d3a3f9fb9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/dual_alpha12_seed_5 | ensemble | validation | 50 | 90.38 | 90.39 | [파일](raw/main/b06d3a3f9fb9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/minor_only_alpha12_seed_2 | minor | validation | 50 | 88.24 | 88.32 | [파일](raw/main/2b1135b968c0/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/minor_only_alpha12_seed_3 | minor | validation | 50 | 89.42 | 89.50 | [파일](raw/main/acff7e2e2874/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/minor_only_alpha12_seed_4 | minor | validation | 50 | 87.94 | 87.98 | [파일](raw/main/e10b040badb0/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/alpha12_dual_minor_seed1_5_ratio100_e50_s512/minor_only_alpha12_seed_5 | minor | validation | 50 | 89.66 | 89.70 | [파일](raw/main/02a9f2dc2e79/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_1 | major | validation | 50 | 86.20 | 86.34 | [파일](raw/main/ecee878b59eb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_1 | minor | validation | 50 | 88.36 | 88.50 | [파일](raw/main/ecee878b59eb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_1 | ensemble | validation | 50 | 88.48 | 88.61 | [파일](raw/main/ecee878b59eb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_2 | major | validation | 50 | 85.40 | 85.54 | [파일](raw/main/ad26f76decb7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_2 | minor | validation | 50 | 88.16 | 88.27 | [파일](raw/main/ad26f76decb7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_2 | ensemble | validation | 50 | 88.12 | 88.23 | [파일](raw/main/ad26f76decb7/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_3 | major | validation | 50 | 83.66 | 83.81 | [파일](raw/main/5ce5b5cb6a11/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_3 | minor | validation | 50 | 87.56 | 87.67 | [파일](raw/main/5ce5b5cb6a11/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_3 | ensemble | validation | 50 | 87.50 | 87.60 | [파일](raw/main/5ce5b5cb6a11/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_4 | major | validation | 50 | 80.98 | 81.05 | [파일](raw/main/283d987ddd27/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_4 | minor | validation | 50 | 85.84 | 86.00 | [파일](raw/main/283d987ddd27/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/bias_strength_valsplit_ratio100_seed1_e50_s512/bias_4 | ensemble | validation | 50 | 85.78 | 85.94 | [파일](raw/main/283d987ddd27/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_no_cross_validation_cifar10_e50_s512_seed1 | major | validation | 50 | 79.12 | 79.01 | [파일](raw/main/25048bb887f1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_no_cross_validation_cifar10_e50_s512_seed1 | minor | validation | 50 | 88.42 | 88.54 | [파일](raw/main/25048bb887f1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_no_cross_validation_cifar10_e50_s512_seed1 | ensemble | validation | 50 | 87.92 | 88.04 | [파일](raw/main/25048bb887f1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_no_cross_validation_cifar10_e50_s512_seed1 | minor_lla | validation | 50 | 87.84 | 87.93 | [파일](raw/main/25048bb887f1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_no_cross_validation_cifar10_e50_s512_seed1 | ensemble_lla | validation | 50 | 86.96 | 87.08 | [파일](raw/main/25048bb887f1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed1 | major | official_test | 50 | 86.08 | 86.20 | [파일](raw/main/0d76d6a35f28/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed1 | minor | official_test | 50 | 88.88 | 88.93 | [파일](raw/main/0d76d6a35f28/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed1 | minor_lla | official_test | 50 | 89.40 | 89.45 | [파일](raw/main/0d76d6a35f28/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed1 | ensemble_lla | official_test | 50 | 89.02 | 89.08 | [파일](raw/main/0d76d6a35f28/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed1 | ensemble | official_test | 50 | 88.66 | 88.72 | [파일](raw/main/0d76d6a35f28/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed2 | major | official_test | 50 | 86.14 | 86.26 | [파일](raw/main/2a31b01881b4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed2 | minor | official_test | 50 | 89.11 | 89.18 | [파일](raw/main/2a31b01881b4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed2 | minor_lla | official_test | 50 | 89.27 | 89.35 | [파일](raw/main/2a31b01881b4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed2 | ensemble_lla | official_test | 50 | 88.93 | 89.02 | [파일](raw/main/2a31b01881b4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_official_test_e50_seed2 | ensemble | official_test | 50 | 88.58 | 88.66 | [파일](raw/main/2a31b01881b4/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1 | major | validation | 50 | 86.80 | 86.88 | [파일](raw/main/a8b4b6443300/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1 | minor | validation | 50 | 89.16 | 89.26 | [파일](raw/main/a8b4b6443300/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1 | ensemble | validation | 50 | 89.02 | 89.11 | [파일](raw/main/a8b4b6443300/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1 | minor_lla | validation | 50 | 89.36 | 89.47 | [파일](raw/main/a8b4b6443300/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1 | ensemble_lla | validation | 50 | 89.42 | 89.52 | [파일](raw/main/a8b4b6443300/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed2 | major | validation | 50 | 85.94 | 85.99 | [파일](raw/main/ad6a1e0d3a2f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed2 | minor | validation | 50 | 89.06 | 89.13 | [파일](raw/main/ad6a1e0d3a2f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed2 | ensemble | validation | 50 | 88.42 | 88.47 | [파일](raw/main/ad6a1e0d3a2f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed2 | minor_lla | validation | 50 | 89.34 | 89.42 | [파일](raw/main/ad6a1e0d3a2f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed2 | ensemble_lla | validation | 50 | 88.76 | 88.82 | [파일](raw/main/ad6a1e0d3a2f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_confidence_validation_cifar10_e50_s512_seed1 | major | validation | 50 | 86.90 | 87.01 | [파일](raw/main/ec52a83f5387/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_confidence_validation_cifar10_e50_s512_seed1 | minor | validation | 50 | 88.94 | 89.01 | [파일](raw/main/ec52a83f5387/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_confidence_validation_cifar10_e50_s512_seed1 | ensemble | validation | 50 | 88.44 | 88.52 | [파일](raw/main/ec52a83f5387/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_confidence_validation_cifar10_e50_s512_seed1 | minor_lla | validation | 50 | 89.06 | 89.15 | [파일](raw/main/ec52a83f5387/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_confidence_validation_cifar10_e50_s512_seed1 | ensemble_lla | validation | 50 | 89.02 | 89.11 | [파일](raw/main/ec52a83f5387/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_floor08_validation_cifar10_e50_s512_seed1 | major | validation | 50 | 86.38 | 86.50 | [파일](raw/main/7e35a7c85465/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_floor08_validation_cifar10_e50_s512_seed1 | minor | validation | 50 | 88.26 | 88.35 | [파일](raw/main/7e35a7c85465/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_floor08_validation_cifar10_e50_s512_seed1 | ensemble | validation | 50 | 88.14 | 88.23 | [파일](raw/main/7e35a7c85465/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_floor08_validation_cifar10_e50_s512_seed1 | minor_lla | validation | 50 | 88.56 | 88.67 | [파일](raw/main/7e35a7c85465/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_2_self_floor08_validation_cifar10_e50_s512_seed1 | ensemble_lla | validation | 50 | 88.48 | 88.58 | [파일](raw/main/7e35a7c85465/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed1 | major | validation | 50 | 86.56 | 86.72 | [파일](raw/main/b7f1a98cce7e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed1 | minor | validation | 50 | 88.60 | 88.67 | [파일](raw/main/b7f1a98cce7e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed1 | ensemble | validation | 50 | 88.50 | 88.59 | [파일](raw/main/b7f1a98cce7e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed1 | minor_lla | validation | 50 | 88.90 | 88.99 | [파일](raw/main/b7f1a98cce7e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed1 | ensemble_lla | validation | 50 | 88.84 | 88.94 | [파일](raw/main/b7f1a98cce7e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed2 | major | validation | 50 | 85.66 | 85.72 | [파일](raw/main/447d414d4763/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed2 | minor | validation | 50 | 88.60 | 88.64 | [파일](raw/main/447d414d4763/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed2 | ensemble | validation | 50 | 88.12 | 88.17 | [파일](raw/main/447d414d4763/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed2 | minor_lla | validation | 50 | 89.14 | 89.20 | [파일](raw/main/447d414d4763/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/dual_real_last_lla_soft_validation_cifar10_e50_s512_seed2 | ensemble_lla | validation | 50 | 88.44 | 88.50 | [파일](raw/main/447d414d4763/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p2 | major | unknown | 1 | 85.75 | 85.89 | [파일](raw/main/fe4a89833ecf/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p2 | minor | unknown | 1 | 88.45 | 88.53 | [파일](raw/main/fe4a89833ecf/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p2 | ensemble | unknown | 1 | 86.76 | 86.87 | [파일](raw/main/fe4a89833ecf/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p3 | major | unknown | 1 | 85.75 | 85.89 | [파일](raw/main/8dc6bb4d1c13/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p3 | minor | unknown | 1 | 88.45 | 88.53 | [파일](raw/main/8dc6bb4d1c13/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p3 | ensemble | unknown | 1 | 87.33 | 87.42 | [파일](raw/main/8dc6bb4d1c13/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p4 | major | unknown | 1 | 85.75 | 85.89 | [파일](raw/main/755d1e3a9ef6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p4 | minor | unknown | 1 | 88.45 | 88.53 | [파일](raw/main/755d1e3a9ef6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p4 | ensemble | unknown | 1 | 87.82 | 87.91 | [파일](raw/main/755d1e3a9ef6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p5 | major | unknown | 1 | 85.75 | 85.89 | [파일](raw/main/7d51d95f239e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p5 | minor | unknown | 1 | 88.45 | 88.53 | [파일](raw/main/7d51d95f239e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p5 | ensemble | unknown | 1 | 88.05 | 88.13 | [파일](raw/main/7d51d95f239e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p6 | major | unknown | 1 | 85.75 | 85.89 | [파일](raw/main/8f6b0394e8cf/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p6 | minor | unknown | 1 | 88.45 | 88.53 | [파일](raw/main/8f6b0394e8cf/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p6 | ensemble | unknown | 1 | 88.33 | 88.41 | [파일](raw/main/8f6b0394e8cf/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p7 | major | unknown | 1 | 85.75 | 85.89 | [파일](raw/main/ef652661f368/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p7 | minor | unknown | 1 | 88.45 | 88.53 | [파일](raw/main/ef652661f368/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p7 | ensemble | unknown | 1 | 88.39 | 88.47 | [파일](raw/main/ef652661f368/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p8 | major | unknown | 1 | 85.75 | 85.89 | [파일](raw/main/f6e41a30ab49/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p8 | minor | unknown | 1 | 88.45 | 88.53 | [파일](raw/main/f6e41a30ab49/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/eval_w0p8 | ensemble | unknown | 1 | 88.52 | 88.60 | [파일](raw/main/f6e41a30ab49/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/train_dual | major | unknown | 50 | 85.75 | 85.89 | [파일](raw/main/5ff1c41649f2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/train_dual | minor | unknown | 50 | 88.45 | 88.53 | [파일](raw/main/5ff1c41649f2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb100/train_dual | ensemble | unknown | 50 | 88.39 | 88.47 | [파일](raw/main/5ff1c41649f2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p2 | major | unknown | 1 | 86.77 | 86.87 | [파일](raw/main/94714156000b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p2 | minor | unknown | 1 | 88.78 | 88.83 | [파일](raw/main/94714156000b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p2 | ensemble | unknown | 1 | 87.71 | 87.79 | [파일](raw/main/94714156000b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p3 | major | unknown | 1 | 86.77 | 86.87 | [파일](raw/main/847b95cde037/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p3 | minor | unknown | 1 | 88.78 | 88.83 | [파일](raw/main/847b95cde037/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p3 | ensemble | unknown | 1 | 88.04 | 88.10 | [파일](raw/main/847b95cde037/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p4 | major | unknown | 1 | 86.77 | 86.87 | [파일](raw/main/063949cd8f55/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p4 | minor | unknown | 1 | 88.78 | 88.83 | [파일](raw/main/063949cd8f55/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p4 | ensemble | unknown | 1 | 88.35 | 88.41 | [파일](raw/main/063949cd8f55/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p5 | major | unknown | 1 | 86.77 | 86.87 | [파일](raw/main/8e8a3b27cdd1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p5 | minor | unknown | 1 | 88.78 | 88.83 | [파일](raw/main/8e8a3b27cdd1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p5 | ensemble | unknown | 1 | 88.54 | 88.59 | [파일](raw/main/8e8a3b27cdd1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p6 | major | unknown | 1 | 86.77 | 86.87 | [파일](raw/main/c446ae8a663f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p6 | minor | unknown | 1 | 88.78 | 88.83 | [파일](raw/main/c446ae8a663f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p6 | ensemble | unknown | 1 | 88.66 | 88.71 | [파일](raw/main/c446ae8a663f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p7 | major | unknown | 1 | 86.77 | 86.87 | [파일](raw/main/50a22af6cb9e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p7 | minor | unknown | 1 | 88.78 | 88.83 | [파일](raw/main/50a22af6cb9e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p7 | ensemble | unknown | 1 | 88.77 | 88.82 | [파일](raw/main/50a22af6cb9e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p8 | major | unknown | 1 | 86.77 | 86.87 | [파일](raw/main/c2c88d58b712/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p8 | minor | unknown | 1 | 88.78 | 88.83 | [파일](raw/main/c2c88d58b712/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/eval_w0p8 | ensemble | unknown | 1 | 88.75 | 88.80 | [파일](raw/main/c2c88d58b712/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/train_dual | major | unknown | 50 | 86.77 | 86.87 | [파일](raw/main/56524116fa06/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/train_dual | minor | unknown | 50 | 88.78 | 88.83 | [파일](raw/main/56524116fa06/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/ensemble_weight_sweep_imb50_100_e50_s512_seed5/imb50/train_dual | ensemble | unknown | 50 | 88.77 | 88.82 | [파일](raw/main/56524116fa06/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb100/baseline_fixmatch | fixmatch | unknown | 50 | 76.69 | 76.57 | [파일](raw/main/64e302bf4493/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb100/dual_sampler_weighted_ce | major | unknown | 50 | 85.41 | 85.55 | [파일](raw/main/fb3ce25f5e93/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb100/dual_sampler_weighted_ce | minor | unknown | 50 | 88.09 | 88.20 | [파일](raw/main/fb3ce25f5e93/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb100/dual_sampler_weighted_ce | ensemble | unknown | 50 | 87.95 | 88.05 | [파일](raw/main/fb3ce25f5e93/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb50/baseline_fixmatch | fixmatch | unknown | 50 | 81.56 | 81.70 | [파일](raw/main/e8277fe3cc57/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb50/dual_sampler_weighted_ce | major | unknown | 50 | 86.66 | 86.76 | [파일](raw/main/348f0d298aa6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb50/dual_sampler_weighted_ce | minor | unknown | 50 | 88.77 | 88.83 | [파일](raw/main/348f0d298aa6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/fixmatch_vs_dual_imb50_100_e50_s512_seed5/imb50/dual_sampler_weighted_ce | ensemble | unknown | 50 | 88.55 | 88.61 | [파일](raw/main/348f0d298aa6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last2_dual_validation_cifar10_e50_s512_seed1_20261023 | major | validation | 50 | 87.32 | 87.42 | [파일](raw/main/2eaebe299888/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last2_dual_validation_cifar10_e50_s512_seed1_20261023 | minor | validation | 50 | 90.08 | 90.10 | [파일](raw/main/2eaebe299888/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last2_dual_validation_cifar10_e50_s512_seed1_20261023 | ensemble | validation | 50 | 89.78 | 89.81 | [파일](raw/main/2eaebe299888/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last_dual_quick_validation_cifar10_e20_s512_seed1_20261023 | major | validation | 20 | 78.58 | 78.85 | [파일](raw/main/11029405247e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last_dual_quick_validation_cifar10_e20_s512_seed1_20261023 | minor | validation | 20 | 82.70 | 82.93 | [파일](raw/main/11029405247e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last_dual_quick_validation_cifar10_e20_s512_seed1_20261023 | ensemble | validation | 20 | 82.48 | 82.73 | [파일](raw/main/11029405247e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last_dual_validation_cifar10_e50_s512_seed1_20261023 | major | validation | 50 | 86.58 | 86.68 | [파일](raw/main/6ec162f61fb6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last_dual_validation_cifar10_e50_s512_seed1_20261023 | minor | validation | 50 | 89.08 | 89.14 | [파일](raw/main/6ec162f61fb6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/last_dual_validation_cifar10_e50_s512_seed1_20261023 | ensemble | validation | 50 | 88.96 | 89.02 | [파일](raw/main/6ec162f61fb6/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_125 | major | validation | 50 | 86.14 | 86.28 | [파일](raw/main/2c2184645c8b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_125 | minor | validation | 50 | 87.64 | 87.70 | [파일](raw/main/2c2184645c8b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_125 | ensemble | validation | 50 | 88.86 | 88.95 | [파일](raw/main/2c2184645c8b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_130 | major | validation | 50 | 86.34 | 86.46 | [파일](raw/main/4d5ab9439599/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_130 | minor | validation | 50 | 88.16 | 88.18 | [파일](raw/main/4d5ab9439599/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_130 | ensemble | validation | 50 | 88.98 | 89.03 | [파일](raw/main/4d5ab9439599/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_135 | major | validation | 50 | 86.58 | 86.71 | [파일](raw/main/2840cb895ead/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_135 | minor | validation | 50 | 87.96 | 88.00 | [파일](raw/main/2840cb895ead/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_135 | ensemble | validation | 50 | 88.90 | 88.98 | [파일](raw/main/2840cb895ead/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_140 | major | validation | 50 | 86.82 | 86.94 | [파일](raw/main/aac93f59818c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_140 | minor | validation | 50 | 86.64 | 86.51 | [파일](raw/main/aac93f59818c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_140 | ensemble | validation | 50 | 88.92 | 88.97 | [파일](raw/main/aac93f59818c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_145 | major | validation | 35 | 82.92 | 83.08 | [파일](raw/main/2c3447a0213e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_145 | minor | validation | 35 | 83.02 | 82.77 | [파일](raw/main/2c3447a0213e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/dual_alpha_145 | ensemble | validation | 35 | 85.98 | 86.05 | [파일](raw/main/2c3447a0213e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/minor_alpha12_then_dual_alpha125_150_ratio100_seed1_e50_s512/minor_only_alpha_12 | minor | validation | 50 | 88.94 | 88.98 | [파일](raw/main/499aebef65af/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_conflict_filter_ratio100_seed1_e50_s512 | major | validation | 50 | 86.90 | 87.03 | [파일](raw/main/9f04acb0a53b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_conflict_filter_ratio100_seed1_e50_s512 | minor | validation | 50 | 88.64 | 88.69 | [파일](raw/main/9f04acb0a53b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_conflict_filter_ratio100_seed1_e50_s512 | ensemble | validation | 50 | 88.60 | 88.66 | [파일](raw/main/9f04acb0a53b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_2_3_e50_s512/seed_2 | major | validation | 50 | 86.32 | 86.40 | [파일](raw/main/81e5a202b165/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_2_3_e50_s512/seed_2 | minor | validation | 50 | 88.40 | 88.45 | [파일](raw/main/81e5a202b165/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_2_3_e50_s512/seed_2 | ensemble | validation | 50 | 88.54 | 88.59 | [파일](raw/main/81e5a202b165/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_2_3_e50_s512/seed_3 | major | validation | 50 | 87.56 | 87.62 | [파일](raw/main/18fe4bac41c9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_2_3_e50_s512/seed_3 | minor | validation | 50 | 89.34 | 89.39 | [파일](raw/main/18fe4bac41c9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_2_3_e50_s512/seed_3 | ensemble | validation | 50 | 89.58 | 89.64 | [파일](raw/main/18fe4bac41c9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_e50_s512 | major | validation | 50 | 86.52 | 86.63 | [파일](raw/main/a2acf51bcdc9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_e50_s512 | minor | validation | 50 | 88.50 | 88.56 | [파일](raw/main/a2acf51bcdc9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_head_tail_selective_cross_ratio100_seed1_e50_s512 | ensemble | validation | 50 | 88.46 | 88.53 | [파일](raw/main/a2acf51bcdc9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_sampler_alpha12_ratio100_seed1_e50_s512 | major | validation | 50 | 87.36 | 87.44 | [파일](raw/main/bc02505d6333/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_sampler_alpha12_ratio100_seed1_e50_s512 | minor | validation | 50 | 89.58 | 89.66 | [파일](raw/main/bc02505d6333/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_sampler_alpha12_ratio100_seed1_e50_s512 | ensemble | validation | 50 | 89.38 | 89.46 | [파일](raw/main/bc02505d6333/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf08_ratio100_seed1_e50_s512 | major | validation | 3 | 20.16 | 12.49 | [파일](raw/main/c8095aef4c9f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf08_ratio100_seed1_e50_s512 | minor | validation | 3 | 42.54 | 41.60 | [파일](raw/main/c8095aef4c9f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf08_ratio100_seed1_e50_s512 | ensemble | validation | 3 | 42.20 | 41.64 | [파일](raw/main/c8095aef4c9f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf08_ratio100_seed1_e50_s512_gpu | major | validation | 50 | 86.50 | 86.63 | [파일](raw/main/4bba951e495b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf08_ratio100_seed1_e50_s512_gpu | minor | validation | 50 | 86.64 | 86.73 | [파일](raw/main/4bba951e495b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf08_ratio100_seed1_e50_s512_gpu | ensemble | validation | 50 | 87.20 | 87.29 | [파일](raw/main/4bba951e495b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep093_098_ratio100_seed1_e50_s512/thr_093 | major | validation | 24 | 79.46 | 79.69 | [파일](raw/main/98ef0a83f1a9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep093_098_ratio100_seed1_e50_s512/thr_093 | minor | validation | 24 | 83.10 | 83.32 | [파일](raw/main/98ef0a83f1a9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep093_098_ratio100_seed1_e50_s512/thr_093 | ensemble | validation | 24 | 83.18 | 83.40 | [파일](raw/main/98ef0a83f1a9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_06 | major | validation | 50 | 85.48 | 85.64 | [파일](raw/main/2038e322c3e9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_06 | minor | validation | 50 | 84.64 | 84.75 | [파일](raw/main/2038e322c3e9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_06 | ensemble | validation | 50 | 85.60 | 85.72 | [파일](raw/main/2038e322c3e9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_07 | major | validation | 50 | 85.42 | 85.53 | [파일](raw/main/1605883b197b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_07 | minor | validation | 50 | 86.60 | 86.70 | [파일](raw/main/1605883b197b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_07 | ensemble | validation | 50 | 86.80 | 86.92 | [파일](raw/main/1605883b197b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_08 | major | validation | 50 | 86.22 | 86.38 | [파일](raw/main/9e539a02eaae/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_08 | minor | validation | 50 | 86.50 | 86.57 | [파일](raw/main/9e539a02eaae/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_08 | ensemble | validation | 50 | 87.18 | 87.27 | [파일](raw/main/9e539a02eaae/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_09 | major | validation | 50 | 86.46 | 86.59 | [파일](raw/main/010a82a68786/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_09 | minor | validation | 50 | 87.60 | 87.68 | [파일](raw/main/010a82a68786/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/new_dual_skip_minor_conf_sweep_ratio100_seed1_e50_s512/thr_09 | ensemble | validation | 50 | 88.10 | 88.17 | [파일](raw/main/010a82a68786/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_decon_dual_matched_cifar10_svhn_e50_s512_seed1_2_3/cifar10/decon/seed_1 | decon | official_test | 50 | 91.11 | 91.09 | [파일](raw/main/2a1435d84341/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_decon_dual_matched_cifar10_svhn_e50_s512_seed1_2_3/cifar10/decon/seed_2 | decon | official_test | 50 | 90.71 | 90.67 | [파일](raw/main/7d20538d0440/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_decon_dual_matched_cifar10_svhn_e50_s512_seed1_2_3/cifar10/decon/seed_3 | decon | official_test | 50 | 90.92 | 90.89 | [파일](raw/main/26d9c990c173/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_decon_dual_matched_cifar10_svhn_e50_s512_seed1_2_3/svhn/decon/seed_1 | decon | official_test | 50 | 96.76 | 96.55 | [파일](raw/main/e7227ec8b3ae/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/decon/seed_1 | decon | official_test | 50 | 83.36 | 83.32 | [파일](raw/main/eabfc4fa5976/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/decon/seed_2 | decon | official_test | 50 | 83.69 | 83.67 | [파일](raw/main/f8c6e0d675f9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/decon/seed_3 | decon | official_test | 50 | 83.76 | 83.73 | [파일](raw/main/d644605fceaf/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_1 | major | official_test | 50 | 86.84 | 86.93 | [파일](raw/main/efe2a0f043fa/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_1 | minor | official_test | 50 | 89.29 | 89.34 | [파일](raw/main/efe2a0f043fa/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_1 | ensemble | official_test | 50 | 89.07 | 89.13 | [파일](raw/main/efe2a0f043fa/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_2 | major | official_test | 50 | 86.06 | 86.21 | [파일](raw/main/c9495d17a5d1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_2 | minor | official_test | 50 | 88.71 | 88.77 | [파일](raw/main/c9495d17a5d1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_2 | ensemble | official_test | 50 | 88.54 | 88.62 | [파일](raw/main/c9495d17a5d1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_3 | major | official_test | 50 | 86.65 | 86.74 | [파일](raw/main/d78688a05c68/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_3 | minor | official_test | 50 | 88.93 | 88.94 | [파일](raw/main/d78688a05c68/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_3 | ensemble | official_test | 50 | 88.82 | 88.84 | [파일](raw/main/d78688a05c68/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/minor/seed_1 | minor | official_test | 50 | 88.52 | 88.59 | [파일](raw/main/28deb2e0e737/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/minor/seed_2 | minor | official_test | 50 | 88.91 | 88.95 | [파일](raw/main/a80828a0084d/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/minor/seed_3 | minor | official_test | 50 | 88.11 | 88.16 | [파일](raw/main/8a2ac2280ce5/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/decon/seed_1 | decon | official_test | 50 | 91.96 | 90.78 | [파일](raw/main/c7896e63f38e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/decon/seed_2 | decon | official_test | 50 | 92.45 | 91.23 | [파일](raw/main/ad62c0e0a1ab/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/decon/seed_3 | decon | official_test | 50 | 92.27 | 91.00 | [파일](raw/main/c8bfd9fc7cf1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_1 | major | official_test | 50 | 94.81 | 94.11 | [파일](raw/main/f40869f48368/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_1 | minor | official_test | 50 | 95.49 | 94.93 | [파일](raw/main/f40869f48368/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_1 | ensemble | official_test | 50 | 95.47 | 94.92 | [파일](raw/main/f40869f48368/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_2 | major | official_test | 50 | 95.45 | 95.04 | [파일](raw/main/b981541205a2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_2 | minor | official_test | 50 | 95.97 | 95.58 | [파일](raw/main/b981541205a2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_2 | ensemble | official_test | 50 | 95.99 | 95.60 | [파일](raw/main/b981541205a2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_3 | major | official_test | 50 | 94.76 | 94.13 | [파일](raw/main/969de064a3eb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_3 | minor | official_test | 50 | 95.45 | 94.87 | [파일](raw/main/969de064a3eb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/dual/seed_3 | ensemble | official_test | 50 | 95.43 | 94.87 | [파일](raw/main/969de064a3eb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/minor/seed_1 | minor | official_test | 50 | 94.94 | 94.62 | [파일](raw/main/80c2f071237e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/minor/seed_2 | minor | official_test | 50 | 95.28 | 95.03 | [파일](raw/main/7d5b57f8624f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_alpha12_cifar10_svhn10_3models_e50_s512_seed1_2_3/svhn/minor/seed_3 | minor | official_test | 50 | 94.73 | 94.41 | [파일](raw/main/4c56dfdfcce8/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/decon/seed_1 | decon | official_test | 50 | 83.58 | 83.57 | [파일](raw/main/11b523bd4629/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_1 | major | official_test | 50 | 86.07 | 86.22 | [파일](raw/main/8dda4dd6ac9f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_1 | minor | official_test | 50 | 88.17 | 88.29 | [파일](raw/main/8dda4dd6ac9f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/dual/seed_1 | ensemble | official_test | 50 | 88.44 | 88.54 | [파일](raw/main/8dda4dd6ac9f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/official_test_finalonly_cifar10_svhn10_3models_e50_s512_seed1_2_3/cifar10/minor/seed_1 | minor | official_test | 50 | 88.68 | 88.75 | [파일](raw/main/390fb4a8bbbc/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/reduced10pct_decon_real_last2_cifar10_seed1/decon | decon | official_test | 50 | 53.85 | 51.22 | [파일](raw/main/6828cf7a68fb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/reduced10pct_decon_real_last2_cifar10_seed1/real_last_dual_2 | major | official_test | 50 | 63.18 | 62.47 | [파일](raw/main/01e4b3c2e14b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/reduced10pct_decon_real_last2_cifar10_seed1/real_last_dual_2 | minor | official_test | 50 | 65.67 | 65.96 | [파일](raw/main/01e4b3c2e14b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/reduced10pct_decon_real_last2_cifar10_seed1/real_last_dual_2 | ensemble | official_test | 50 | 65.91 | 66.20 | [파일](raw/main/01e4b3c2e14b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/reduced10pct_decon_real_last2_cifar10_seed1/real_last_dual_2 | minor_lla | official_test | 50 | 67.50 | 67.57 | [파일](raw/main/01e4b3c2e14b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/reduced10pct_decon_real_last2_cifar10_seed1/real_last_dual_2 | ensemble_lla | official_test | 50 | 67.35 | 67.49 | [파일](raw/main/01e4b3c2e14b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_dual_minor_decon_seed1_e50_s512/decon_workers0 | decon | validation | 50 | 91.96 | 91.94 | [파일](raw/main/2aa6d4abfc5c/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_dual_minor_decon_seed1_e50_s512/dual_alpha12_verified | major | validation | 50 | 94.48 | 94.50 | [파일](raw/main/80c9d9b74113/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_dual_minor_decon_seed1_e50_s512/dual_alpha12_verified | minor | validation | 50 | 95.02 | 95.02 | [파일](raw/main/80c9d9b74113/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_dual_minor_decon_seed1_e50_s512/dual_alpha12_verified | ensemble | validation | 50 | 94.94 | 94.95 | [파일](raw/main/80c9d9b74113/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_dual_minor_decon_seed1_e50_s512/minor_only_alpha12_direct | minor | validation | 50 | 94.30 | 94.30 | [파일](raw/main/64d5af245885/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/decon_seed_2 | decon | validation | 50 | 91.38 | 91.35 | [파일](raw/main/f54c67357f0b/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/decon_seed_3 | decon | validation | 50 | 91.70 | 91.70 | [파일](raw/main/174e36315c36/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/decon_seed_4 | decon | validation | 50 | 91.88 | 91.87 | [파일](raw/main/6a661e1ab5e1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/decon_seed_5 | decon | validation | 50 | 91.54 | 91.52 | [파일](raw/main/655b63424249/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_2_direct | major | validation | 50 | 93.30 | 93.31 | [파일](raw/main/82d298bbbabb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_2_direct | minor | validation | 50 | 93.46 | 93.47 | [파일](raw/main/82d298bbbabb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_2_direct | ensemble | validation | 50 | 93.66 | 93.66 | [파일](raw/main/82d298bbbabb/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_3 | major | validation | 50 | 94.30 | 94.34 | [파일](raw/main/ed897aa4d919/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_3 | minor | validation | 50 | 94.84 | 94.85 | [파일](raw/main/ed897aa4d919/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_3 | ensemble | validation | 50 | 94.80 | 94.82 | [파일](raw/main/ed897aa4d919/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_4 | major | validation | 50 | 94.86 | 94.88 | [파일](raw/main/d4d28e436336/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_4 | minor | validation | 50 | 95.34 | 95.35 | [파일](raw/main/d4d28e436336/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_4 | ensemble | validation | 50 | 95.46 | 95.47 | [파일](raw/main/d4d28e436336/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_5 | major | validation | 50 | 94.38 | 94.41 | [파일](raw/main/e7898f4de885/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_5 | minor | validation | 50 | 95.10 | 95.11 | [파일](raw/main/e7898f4de885/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/dual_alpha12_seed_5 | ensemble | validation | 50 | 95.26 | 95.27 | [파일](raw/main/e7898f4de885/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/minor_only_alpha12_seed_2 | minor | validation | 50 | 93.04 | 93.00 | [파일](raw/main/1840cb741639/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/minor_only_alpha12_seed_3 | minor | validation | 50 | 93.78 | 93.80 | [파일](raw/main/05ae1c87394f/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/minor_only_alpha12_seed_4 | minor | validation | 50 | 94.20 | 94.21 | [파일](raw/main/62802ee3f6f2/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/svhn_lt_ratio100_seed1_5_e50_s512_optimized_v2/minor_only_alpha12_seed_5 | minor | validation | 50 | 93.88 | 93.89 | [파일](raw/main/8bd24b6703b1/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/decon | decon | official_test | 50 | 90.72 | 90.69 | [파일](raw/main/a17365686814/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/new_dual | major | official_test | 50 | 86.60 | 86.70 | [파일](raw/main/af6d48b10532/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/new_dual | minor | official_test | 50 | 89.06 | 89.12 | [파일](raw/main/af6d48b10532/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/new_dual | ensemble | official_test | 50 | 88.80 | 88.88 | [파일](raw/main/af6d48b10532/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/real_last_dual_2 | major | official_test | 50 | 86.16 | 86.26 | [파일](raw/main/31fd65a22cf9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/real_last_dual_2 | minor | official_test | 50 | 89.12 | 89.18 | [파일](raw/main/31fd65a22cf9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/real_last_dual_2 | ensemble | official_test | 50 | 88.76 | 88.82 | [파일](raw/main/31fd65a22cf9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/real_last_dual_2 | minor_lla | official_test | 50 | 89.43 | 89.49 | [파일](raw/main/31fd65a22cf9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/three_models_matched_fresh_cifar10_seed1/real_last_dual_2 | ensemble_lla | official_test | 50 | 89.29 | 89.36 | [파일](raw/main/31fd65a22cf9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/decon/seed_1 | decon | validation | 50 | 81.86 | 81.86 | [파일](raw/main/e3d57a73abf9/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/decon/seed_2 | decon | validation | 50 | 82.18 | 82.19 | [파일](raw/main/bc12ba6eeb0e/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/decon/seed_3 | decon | validation | 50 | 82.80 | 82.82 | [파일](raw/main/456cd5a356a3/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_1 | major | validation | 50 | 86.30 | 86.42 | [파일](raw/main/bd0de3372f22/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_1 | minor | validation | 50 | 88.28 | 88.40 | [파일](raw/main/bd0de3372f22/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_1 | ensemble | validation | 50 | 88.12 | 88.24 | [파일](raw/main/bd0de3372f22/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_2 | major | validation | 50 | 85.62 | 85.80 | [파일](raw/main/e9f17ea84234/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_2 | minor | validation | 50 | 87.60 | 87.70 | [파일](raw/main/e9f17ea84234/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_2 | ensemble | validation | 50 | 87.86 | 87.94 | [파일](raw/main/e9f17ea84234/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_3 | major | validation | 50 | 86.80 | 86.93 | [파일](raw/main/8b1c0de96ece/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_3 | minor | validation | 50 | 89.50 | 89.61 | [파일](raw/main/8b1c0de96ece/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_ours_vs_decon_e50_s512_seed1_2_3/ours/seed_3 | ensemble | validation | 50 | 89.72 | 89.82 | [파일](raw/main/8b1c0de96ece/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_single_minor_e50_s512_seed1_2_3/seed_1 | minor | validation | 50 | 88.72 | 88.82 | [파일](raw/main/799406063aad/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_single_minor_e50_s512_seed1_2_3/seed_2 | minor | validation | 50 | 88.42 | 88.49 | [파일](raw/main/3ff9862e1bdc/per_class_eval.csv) |
| main/Semi-supervised-learning-main/results/valsplit_ratio100_single_minor_e50_s512_seed1_2_3/seed_3 | minor | validation | 50 | 88.92 | 88.99 | [파일](raw/main/d2a5a0f4ae63/per_class_eval.csv) |
| decon/results/decon_compare_imb50_100_seed1_2_3_e50_s512/imb100/seed_1/decon | decon | unknown | 50 | 83.22 | 83.16 | [파일](raw/decon/8752686ea3b6/per_class_eval.csv) |
| decon/results/decon_compare_imb50_100_seed1_2_3_e50_s512/imb100/seed_2/decon | decon | unknown | 50 | 83.08 | 83.05 | [파일](raw/decon/4a2913452c36/per_class_eval.csv) |
| decon/results/decon_compare_imb50_100_seed1_2_3_e50_s512/imb100/seed_3/decon | decon | unknown | 50 | 83.94 | 83.88 | [파일](raw/decon/1e63ed928696/per_class_eval.csv) |
| decon/results/decon_compare_imb50_100_seed1_2_3_e50_s512/imb50/seed_1/decon | decon | unknown | 50 | 85.06 | 85.01 | [파일](raw/decon/42498dc3644f/per_class_eval.csv) |
| decon/results/decon_compare_imb50_100_seed1_2_3_e50_s512/imb50/seed_2/decon | decon | unknown | 50 | 85.13 | 85.09 | [파일](raw/decon/a198fe5d86a4/per_class_eval.csv) |
| decon/results/decon_compare_imb50_100_seed1_2_3_e50_s512/imb50/seed_3/decon | decon | unknown | 50 | 85.49 | 85.44 | [파일](raw/decon/e4d4350db51d/per_class_eval.csv) |
| decon/results/smoke_dual_matched_cifar_seed1 | decon | unknown | 1 | 9.56 | 2.51 | [파일](raw/decon/a7eb7361b881/per_class_eval.csv) |
| decon/results/svhn_smoke_decon | decon | validation | 1 | 10.10 | 2.90 | [파일](raw/decon/ddaa876414df/per_class_eval.csv) |
| thesis/baseline compare/result/seed_5/abc | abc | unknown | 100 | 82.03 | 82.18 | [파일](raw/thesis/61ddd4251f0a/per_class_eval.csv) |
| thesis/baseline compare/result/seed_5/crest | crest | unknown | 100 | 91.19 | 91.20 | [파일](raw/thesis/8d2d4c6e5639/per_class_eval.csv) |
| thesis/baseline compare/result/seed_5/daso | daso | unknown | 100 | 91.57 | 91.58 | [파일](raw/thesis/4f959f97323e/per_class_eval.csv) |
| thesis/baseline compare/result/seed_5/proposed | major | unknown | 100 | 92.50 | 92.53 | [파일](raw/thesis/c0604064b3dc/per_class_eval.csv) |
| thesis/baseline compare/result/seed_5/proposed | minor | unknown | 100 | 93.22 | 93.24 | [파일](raw/thesis/c0604064b3dc/per_class_eval.csv) |
| thesis/baseline compare/result/seed_5/proposed | ensemble | unknown | 100 | 93.47 | 93.48 | [파일](raw/thesis/c0604064b3dc/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/seed_5/abc | abc | unknown | 100 | 82.03 | 82.18 | [파일](raw/thesis/61ddd4251f0a/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/seed_5/crest | crest | unknown | 100 | 91.19 | 91.20 | [파일](raw/thesis/8d2d4c6e5639/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/seed_5/daso | daso | unknown | 100 | 91.57 | 91.58 | [파일](raw/thesis/4f959f97323e/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/seed_5/proposed | major | unknown | 100 | 92.50 | 92.53 | [파일](raw/thesis/c0604064b3dc/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/seed_5/proposed | minor | unknown | 100 | 93.22 | 93.24 | [파일](raw/thesis/c0604064b3dc/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/seed_5/proposed | ensemble | unknown | 100 | 93.47 | 93.48 | [파일](raw/thesis/c0604064b3dc/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/smoke/seed_5/abc | abc | unknown | 1 | 9.28 | 2.42 | [파일](raw/legacy_machine/89439dd365f6/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/smoke/seed_5/crest | crest | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/cc3b047d16b9/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/smoke/seed_5/daso | daso | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/f8f30402dea4/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/smoke/seed_5/proposed | major | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/a765621eec1b/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/smoke/seed_5/proposed | minor | unknown | 1 | 10.00 | 1.82 | [파일](raw/legacy_machine/a765621eec1b/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/baseline compare/result/smoke/seed_5/proposed | ensemble | unknown | 1 | 10.00 | 1.82 | [파일](raw/legacy_machine/a765621eec1b/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_3day/seed_5/fixmatch | fixmatch | unknown | 40 | 78.17 | 78.14 | [파일](raw/legacy_machine/211284e1dcfd/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_3day_test_resume/seed_5/crest | crest | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/cc3b047d16b9/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_smoke/seed_5/crest | crest | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/c976cc14a265/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_smoke2/seed_5/crest | crest | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/9b616a7277ec/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_smoke3/seed_5/abc | abc | unknown | 1 | 9.28 | 2.42 | [파일](raw/legacy_machine/edf8ca6f426a/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_smoke3/seed_5/daso | daso | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/3d9890aa4dba/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_startcheck/seed_5/fixmatch | fixmatch | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/9b413f8db342/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/bc/baseline_compare_startprocess_check/seed_5/fixmatch | fixmatch | unknown | 1 | 9.29 | 2.42 | [파일](raw/legacy_machine/45c1efa48249/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/epoch200_compare/seed_5/00_fixmatch | fixmatch | unknown | 67 | 81.51 | 81.53 | [파일](raw/legacy_machine/2ca732d0c04e/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/00_fixmatch | fixmatch | unknown | 4 | 31.56 | 22.62 | [파일](raw/legacy_machine/e8c70e56b219/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/01_dual_only | major | unknown | 4 | 35.10 | 25.95 | [파일](raw/legacy_machine/06c6c0267eef/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/01_dual_only | minor | unknown | 4 | 34.12 | 24.60 | [파일](raw/legacy_machine/06c6c0267eef/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/01_dual_only | ensemble | unknown | 4 | 34.56 | 25.23 | [파일](raw/legacy_machine/06c6c0267eef/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/02_dual_sampler | major | unknown | 4 | 32.47 | 23.68 | [파일](raw/legacy_machine/47fc0ef606af/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/02_dual_sampler | minor | unknown | 4 | 47.53 | 44.44 | [파일](raw/legacy_machine/47fc0ef606af/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/02_dual_sampler | ensemble | unknown | 4 | 43.80 | 38.88 | [파일](raw/legacy_machine/47fc0ef606af/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/03_dual_class_weight | major | unknown | 4 | 35.81 | 26.80 | [파일](raw/legacy_machine/885e27436038/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/03_dual_class_weight | minor | unknown | 4 | 37.73 | 29.23 | [파일](raw/legacy_machine/885e27436038/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/03_dual_class_weight | ensemble | unknown | 4 | 37.31 | 28.45 | [파일](raw/legacy_machine/885e27436038/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/04_dual_semantic_proxy | major | unknown | 4 | 35.73 | 26.51 | [파일](raw/legacy_machine/c763c86109ab/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/04_dual_semantic_proxy | minor | unknown | 4 | 33.66 | 24.32 | [파일](raw/legacy_machine/c763c86109ab/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/04_dual_semantic_proxy | ensemble | unknown | 4 | 34.54 | 25.20 | [파일](raw/legacy_machine/c763c86109ab/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/05_dual_topk | major | unknown | 4 | 35.79 | 26.09 | [파일](raw/legacy_machine/d8e7bdd5d378/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/05_dual_topk | minor | unknown | 4 | 33.63 | 24.26 | [파일](raw/legacy_machine/d8e7bdd5d378/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/05_dual_topk | ensemble | unknown | 4 | 34.39 | 24.93 | [파일](raw/legacy_machine/d8e7bdd5d378/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/06_dual_sampler_class_weight | major | unknown | 4 | 34.94 | 26.11 | [파일](raw/legacy_machine/f0ece4d96b1d/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/06_dual_sampler_class_weight | minor | unknown | 4 | 48.38 | 46.26 | [파일](raw/legacy_machine/f0ece4d96b1d/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/06_dual_sampler_class_weight | ensemble | unknown | 4 | 47.11 | 43.54 | [파일](raw/legacy_machine/f0ece4d96b1d/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/07_dual_sampler_semantic_proxy | major | unknown | 4 | 33.57 | 24.23 | [파일](raw/legacy_machine/ba053f3b2379/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/07_dual_sampler_semantic_proxy | minor | unknown | 4 | 48.78 | 46.62 | [파일](raw/legacy_machine/ba053f3b2379/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/07_dual_sampler_semantic_proxy | ensemble | unknown | 4 | 45.38 | 41.21 | [파일](raw/legacy_machine/ba053f3b2379/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/08_dual_sampler_topk | major | unknown | 4 | 34.66 | 25.85 | [파일](raw/legacy_machine/df66e577687d/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/08_dual_sampler_topk | minor | unknown | 4 | 47.14 | 43.92 | [파일](raw/legacy_machine/df66e577687d/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/08_dual_sampler_topk | ensemble | unknown | 4 | 43.68 | 38.62 | [파일](raw/legacy_machine/df66e577687d/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/09_dual_class_weight_semantic_proxy | major | unknown | 4 | 34.15 | 25.27 | [파일](raw/legacy_machine/f9fc146e5680/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/09_dual_class_weight_semantic_proxy | minor | unknown | 4 | 38.05 | 29.89 | [파일](raw/legacy_machine/f9fc146e5680/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/09_dual_class_weight_semantic_proxy | ensemble | unknown | 4 | 37.26 | 28.54 | [파일](raw/legacy_machine/f9fc146e5680/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/10_dual_class_weight_topk | major | unknown | 4 | 36.64 | 27.55 | [파일](raw/legacy_machine/04cfe8527aaf/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/10_dual_class_weight_topk | minor | unknown | 4 | 37.97 | 29.74 | [파일](raw/legacy_machine/04cfe8527aaf/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/10_dual_class_weight_topk | ensemble | unknown | 4 | 37.64 | 29.10 | [파일](raw/legacy_machine/04cfe8527aaf/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/11_dual_semantic_proxy_topk | major | unknown | 4 | 35.35 | 26.49 | [파일](raw/legacy_machine/2d21d0151796/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/11_dual_semantic_proxy_topk | minor | unknown | 4 | 32.82 | 23.59 | [파일](raw/legacy_machine/2d21d0151796/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/11_dual_semantic_proxy_topk | ensemble | unknown | 4 | 33.80 | 24.74 | [파일](raw/legacy_machine/2d21d0151796/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/12_dual_sampler_class_weight_semantic_proxy | major | unknown | 4 | 33.59 | 24.25 | [파일](raw/legacy_machine/523bbabfda63/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/12_dual_sampler_class_weight_semantic_proxy | minor | unknown | 4 | 48.75 | 47.21 | [파일](raw/legacy_machine/523bbabfda63/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/12_dual_sampler_class_weight_semantic_proxy | ensemble | unknown | 4 | 47.01 | 43.92 | [파일](raw/legacy_machine/523bbabfda63/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/13_dual_sampler_class_weight_topk | major | unknown | 4 | 34.08 | 25.25 | [파일](raw/legacy_machine/f82d991d61a5/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/13_dual_sampler_class_weight_topk | minor | unknown | 4 | 49.15 | 48.08 | [파일](raw/legacy_machine/f82d991d61a5/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/13_dual_sampler_class_weight_topk | ensemble | unknown | 4 | 47.36 | 44.74 | [파일](raw/legacy_machine/f82d991d61a5/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/14_dual_sampler_semantic_proxy_topk | major | unknown | 4 | 33.57 | 24.64 | [파일](raw/legacy_machine/4ec6f763f6bb/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/14_dual_sampler_semantic_proxy_topk | minor | unknown | 4 | 47.26 | 44.51 | [파일](raw/legacy_machine/4ec6f763f6bb/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/14_dual_sampler_semantic_proxy_topk | ensemble | unknown | 4 | 43.48 | 39.01 | [파일](raw/legacy_machine/4ec6f763f6bb/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/15_dual_class_weight_semantic_proxy_topk | major | unknown | 4 | 36.85 | 27.83 | [파일](raw/legacy_machine/987733805941/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/15_dual_class_weight_semantic_proxy_topk | minor | unknown | 4 | 38.73 | 30.69 | [파일](raw/legacy_machine/987733805941/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/15_dual_class_weight_semantic_proxy_topk | ensemble | unknown | 4 | 38.64 | 30.22 | [파일](raw/legacy_machine/987733805941/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/16_dual_sampler_class_weight_semantic_proxy_topk | major | unknown | 4 | 32.82 | 24.10 | [파일](raw/legacy_machine/ed2d5ee50038/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/16_dual_sampler_class_weight_semantic_proxy_topk | minor | unknown | 4 | 49.81 | 48.95 | [파일](raw/legacy_machine/ed2d5ee50038/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/16_dual_sampler_class_weight_semantic_proxy_topk | ensemble | unknown | 4 | 48.21 | 45.74 | [파일](raw/legacy_machine/ed2d5ee50038/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/17_full_dual_with_agreement_consistency | major | unknown | 4 | 34.30 | 25.76 | [파일](raw/legacy_machine/57f45cd95348/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/17_full_dual_with_agreement_consistency | minor | unknown | 4 | 49.75 | 48.89 | [파일](raw/legacy_machine/57f45cd95348/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/factorial_compare/seed_5/17_full_dual_with_agreement_consistency | ensemble | unknown | 4 | 47.33 | 45.12 | [파일](raw/legacy_machine/57f45cd95348/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/medium_compare/seed_5/01_fixmatch | fixmatch | unknown | 4 | 33.96 | 25.31 | [파일](raw/legacy_machine/3e456f99b880/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/medium_compare/seed_5/02_dual_only | major | unknown | 4 | 34.48 | 25.02 | [파일](raw/legacy_machine/10f2a57b9ad1/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/medium_compare/seed_5/02_dual_only | minor | unknown | 4 | 33.85 | 23.84 | [파일](raw/legacy_machine/10f2a57b9ad1/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/medium_compare/seed_5/02_dual_only | ensemble | unknown | 4 | 34.04 | 24.12 | [파일](raw/legacy_machine/10f2a57b9ad1/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/medium_compare/seed_5/03_full | major | unknown | 4 | 33.13 | 24.46 | [파일](raw/legacy_machine/70f041a6d69e/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/medium_compare/seed_5/03_full | minor | unknown | 4 | 48.17 | 47.43 | [파일](raw/legacy_machine/70f041a6d69e/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/medium_compare/seed_5/03_full | ensemble | unknown | 4 | 46.13 | 43.38 | [파일](raw/legacy_machine/70f041a6d69e/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare/seed_5/01_fixmatch | fixmatch | unknown | 4 | 16.83 | 8.01 | [파일](raw/legacy_machine/87a91464c91a/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare/seed_5/02_dual_only | major | unknown | 4 | 16.85 | 7.66 | [파일](raw/legacy_machine/0056850a5fdd/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare/seed_5/02_dual_only | minor | unknown | 4 | 20.55 | 11.54 | [파일](raw/legacy_machine/0056850a5fdd/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare/seed_5/02_dual_only | ensemble | unknown | 4 | 20.63 | 11.90 | [파일](raw/legacy_machine/0056850a5fdd/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare/seed_5/03_full | major | unknown | 4 | 18.28 | 8.52 | [파일](raw/legacy_machine/31188968a274/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare/seed_5/03_full | minor | unknown | 4 | 24.66 | 19.19 | [파일](raw/legacy_machine/31188968a274/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare/seed_5/03_full | ensemble | unknown | 4 | 20.55 | 11.37 | [파일](raw/legacy_machine/31188968a274/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare_pseudo/seed_5/01_fixmatch | fixmatch | unknown | 4 | 16.87 | 7.49 | [파일](raw/legacy_machine/33e6dc361df3/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare_pseudo/seed_5/02_dual_only | major | unknown | 4 | 17.33 | 8.35 | [파일](raw/legacy_machine/fb940c247ea6/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare_pseudo/seed_5/02_dual_only | minor | unknown | 4 | 21.96 | 14.25 | [파일](raw/legacy_machine/fb940c247ea6/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare_pseudo/seed_5/02_dual_only | ensemble | unknown | 4 | 20.88 | 11.83 | [파일](raw/legacy_machine/fb940c247ea6/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare_pseudo/seed_5/03_full | major | unknown | 4 | 14.84 | 6.40 | [파일](raw/legacy_machine/341a760be117/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare_pseudo/seed_5/03_full | minor | unknown | 4 | 26.45 | 20.13 | [파일](raw/legacy_machine/341a760be117/per_class_eval.csv) |
| legacy_machine/졸업논문 코드/dual 수정/results/quick_compare_pseudo/seed_5/03_full | ensemble | unknown | 4 | 23.35 | 15.54 | [파일](raw/legacy_machine/341a760be117/per_class_eval.csv) |
| dual200/. | major | unknown | 200 | 93.91 | 93.89 | [파일](raw/dual200/cdb4ee2aea69/per_class_eval.csv.gz) |
| dual200/. | minor | unknown | 200 | 93.74 | 93.70 | [파일](raw/dual200/cdb4ee2aea69/per_class_eval.csv.gz) |
| dual200/. | ensemble | unknown | 200 | 94.09 | 94.07 | [파일](raw/dual200/cdb4ee2aea69/per_class_eval.csv.gz) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed1_e50_s512 | major | unknown | 50 | 86.11 | 86.27 | [파일](raw/onedrive_dual/82256bb959e8/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed1_e50_s512 | minor | unknown | 50 | 87.12 | 88.19 | [파일](raw/onedrive_dual/82256bb959e8/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed1_e50_s512 | ensemble | unknown | 50 | 88.80 | 88.91 | [파일](raw/onedrive_dual/82256bb959e8/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed2_e50_s512 | major | unknown | 50 | 85.69 | 85.86 | [파일](raw/onedrive_dual/69d9f4efa815/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed2_e50_s512 | minor | unknown | 50 | 88.00 | 88.12 | [파일](raw/onedrive_dual/69d9f4efa815/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed2_e50_s512 | ensemble | unknown | 50 | 88.08 | 88.20 | [파일](raw/onedrive_dual/69d9f4efa815/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed3_e50_s512 | major | unknown | 50 | 85.67 | 85.78 | [파일](raw/onedrive_dual/a18bd9b9c33c/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed3_e50_s512 | minor | unknown | 50 | 88.27 | 88.37 | [파일](raw/onedrive_dual/a18bd9b9c33c/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb100_seed3_e50_s512 | ensemble | unknown | 50 | 88.34 | 88.42 | [파일](raw/onedrive_dual/a18bd9b9c33c/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed1_e50_s512 | major | unknown | 50 | 87.26 | 87.37 | [파일](raw/onedrive_dual/2d6510797a72/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed1_e50_s512 | minor | unknown | 50 | 89.22 | 89.27 | [파일](raw/onedrive_dual/2d6510797a72/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed1_e50_s512 | ensemble | unknown | 50 | 89.33 | 89.39 | [파일](raw/onedrive_dual/2d6510797a72/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed2_e50_s512 | major | unknown | 50 | 87.27 | 87.38 | [파일](raw/onedrive_dual/35727cb7b917/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed2_e50_s512 | minor | unknown | 50 | 88.97 | 89.03 | [파일](raw/onedrive_dual/35727cb7b917/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed2_e50_s512 | ensemble | unknown | 50 | 89.04 | 89.10 | [파일](raw/onedrive_dual/35727cb7b917/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed3_e50_s512 | major | unknown | 50 | 86.82 | 86.93 | [파일](raw/onedrive_dual/cc607db926b0/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed3_e50_s512 | minor | unknown | 50 | 88.94 | 89.01 | [파일](raw/onedrive_dual/cc607db926b0/per_class_eval.csv) |
| onedrive_dual/results/minor_weighted_balanced_e50_s512_seed1_2_3/imb50_seed3_e50_s512 | ensemble | unknown | 50 | 88.93 | 89.00 | [파일](raw/onedrive_dual/cc607db926b0/per_class_eval.csv) |
