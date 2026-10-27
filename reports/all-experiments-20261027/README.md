# 전체 실험 기록 — 2026-10-27 정리

이 문서는 로컬에서 확인할 수 있는 **전체 실험 결과**를 한곳에 모은 인덱스입니다.
초기 FixMatch/ablation, Dual·Minor-only, sampler/교환/보정 sweep, CIFAR-10·SVHN,
DeCon 비교, last dual, last-2 dual, LLA soft/hard, self loss 실험, 축소 데이터 실험을 포함합니다.
성공한 결과뿐 아니라 중단·부분 실행·진단·실행 설정도 보존합니다.

## 전체 원본과 실행 목록

- [전체 모델별 최종 기록](CATALOG.md): 실행별 마지막 기록, 정확도, Macro-F1, 원본 링크
- [기계 판독용 성능 표](final_metrics.csv): 그룹별 재현율 포함
- [실행 인벤토리](run_inventory.csv): 마지막 epoch, 확인 가능한 예정 epoch, 상태
- [전체 파일 원본 대응표](artifact_manifest.csv): 원래 경로, SHA256, 압축/중복 여부, 보관 경로
- [아카이브 규모](archive_stats.json)
- [파싱 참고사항](parse_notes.json)
- [로컬에만 남겨둔 대형 파일 목록](local_only_artifacts.csv): 체크포인트·TensorBoard 파일

`raw/`의 짧은 폴더 ID는 Windows 경로 길이 제한을 피하기 위한 것입니다.
실험 이름과 원래 경로는 `artifact_manifest.csv`에서 찾을 수 있습니다.
동일한 내용의 복제 파일은 한 번만 보관하며 모든 원본 경로가 대응표에 남습니다.
250KB 초과 파일은 `.gz`로 무손실 압축했습니다. 압축 해제한 모든 파일의 SHA256를 원본과 검증했습니다.
학습 이미지·가중치·TensorBoard 바이너리는 저장소 용량 때문에 업로드하지 않았습니다.
성능 지표, 클래스별 결과, 샘플별 진단, 데이터 인덱스, 설정과 로그는 포함합니다.

## 읽을 때 주의할 구분

1. **Validation과 official test를 섞어 비교하지 않습니다.** `unknown`은 평가 분할의 근거가 부족한 기록입니다.
2. 전체 train 학습과 validation5000장을 제외한 학습은 서로 다른 조건입니다.
3. DeCon의 초기 불균형 unlabeled 실험과 이후 **Dual-matched 균형 unlabeled 실험은 다릅니다.**
4. seed1·2·3·5회 평균을 섞지 않습니다. 최근 ablation과 10% 데이터 실험은 seed1 한 번입니다.
5. `latest_record_only`는 마지막 기록이 있다는 뜻이지 계획한 학습 완료를 보장하지 않습니다.
6. 과거 보고서도 원본 그대로 보관했습니다. 수정된 해석은 이 문서와 아래 정정 내용을 우선합니다.

## 실험 흐름

| 단계 | 주요 내용 | 기록 위치/검색어 |
|---|---|---|
| 초기 baseline | FixMatch, ABC, CReST 등, 불균형·epoch 비교 | `baseline compare`, `fixmatch`, `ablation`, `dual200` |
| 기존 Dual/Minor-only | balanced sampler α sweep, seed1~5, CIFAR-10/SVHN | `alpha12`, `minor_only`, `svhn_lt` |
| new dual | Head/Tail 교환, confidence skip, conflict filter, ensemble sweep | `new_dual`, `ensemble_weight_sweep` |
| DeCon 비교 | 초기 설정 및 데이터 조건을 맞춘 재실험 | `official_test_*`, `valsplit_ratio100_ours_vs_decon` |
| last dual | 20/50 epoch, 보정 전후·불일치 분석 | `last_dual_*` |
| last-2 dual | 특정 Head→Tail 보정 충돌에서 Minor self 억제 | `last2_dual_*` |
| LLA soft | Minor에 학습형 보정+soft self | `dual_real_last_lla_soft_*` |
| LLA hard | LLA soft 학습 유지, CNN self만 hard | `dual_real_last_2_lla_hard_*` |
| 교환 OFF | 양방향 교환 제거+교환에 따른 self 억제 복원 | `*_no_cross_*` |
| self confidence | 선택된 Minor self CE에 confidence 곱하기 | `*_self_confidence_*` |
| self threshold | 0.95 시작, 최저값0.5→0.8 | `*_self_floor08_*` |
| 같은 조건 새 학습 | new dual/real last-2/DeCon, seed1, 전체 train | `three_models_matched_fresh_cifar10_seed1` |
| 10% 데이터 | L400/U5000, real last-2/DeCon | `reduced10pct_decon_real_last2_cifar10_seed1` |

## A. 최근 동일 조건 전체 데이터 실험 — 공식 test

세 방법 모두 **처음부터 학습**했습니다. CIFAR-10 train50000, labeled4000,
unlabeled50000(균형, labeled 포함), seed1, WRN28-2,50×512 steps, batch32,mu7,
lr.03,wd.0005,EMA.999. Validation 별도 분리 없이 마지막 epoch에 official test10000 평가.
Dual ensemble은 Minor 비중.8. 모델 고유 sampler·loss·구조는 유지했습니다.

| 모델/출력 | 정확도 | Macro-F1 | Tail 재현율 |
|---|---:|---:|---:|
| DeCon | **90.72** | **90.69** | **94.27** |
| real last-2 Minor raw | 89.12 | 89.18 | 89.30 |
| real last-2 Minor LLA | 89.43 | 89.49 | 90.50 |
| real last-2 ensemble raw | 88.76 | 88.82 | — |
| real last-2 ensemble LLA | 89.29 | 89.36 | — |
| new dual Minor | 89.06 | 89.12 | 89.00 |
| new dual ensemble | 88.80 | 88.88 | — |

DeCon과 LLA Minor의 전체 정답 차이129장 중 Tail에서113장 차이.
말 재현율92.3 vs86.3. LLA 모델의 보정 전 성능은 new dual과 거의 같았습니다.

## B. 최근 10% 데이터 실험 — 공식 test

Labeled400, balanced unlabeled5000(클래스당500), 같은 이미지, seed1,50×512 steps.
기존 real last-2의 **self 하한.5/가중치1/cross ON** 사용. 기존 체크포인트 재사용 없음.
Labeled 실제 개수는 **162,97,58,35,21,13,7,4,2,1**입니다.
원래 ratio100 생성 규칙의 정수 배분 때문에 실제 최대/최소는162:1입니다.

| 모델/출력 | 정확도 | Macro-F1 | Tail 재현율 |
|---|---:|---:|---:|
| DeCon | 53.85 | 51.22 | 31.67 |
| real last-2 Minor raw | 65.67 | 65.96 | 73.93 |
| real last-2 Minor LLA | **67.50** | **67.57** | **77.00** |
| real last-2 ensemble LLA | 67.35 | 67.49 | 77.73 |

**추가 분석:** DeCon 트럭 재현율5.6%, LLA Minor89.2%.
DeCon balanced head raw 정확도52.69→보정 후53.85; 보정 자체는+1.16%p.
raw 출력은 test10000장 중 트럭을9번 예측, 실제 트럭1000장 중2장 정답.
보정 후 실제 트럭842장을 자동차로 오분류. 예측 이동평균의 트럭 비중.80%, 실제 pool 비중10%.
분포 편향은 확인됐지만 분포 추정이 학습 실패의 원인인지 결과인지는 분리하지 못했습니다.
Major→Minor 전달 정확도59.61%, Minor→Major87.06%; 축소 조건에서 교환 OFF 비교는 아직 없습니다.
한 seed·트럭 labeled1장이므로 일반화된 우위 주장이 아닌 예비 결과입니다.

## C. Validation ablation — 이전 별도 학습, seed1

공식 train에서5000장(클래스당500)을 제외하고 나머지45000장을 학습용 풀로 사용.
Labeled4000,50 epoch, ensemble Minor 비중.7. 아래는 **validation 결과**입니다.

| 방법 | Minor raw 정확도 | Minor 보정 후 정확도 | 보정 후 Macro-F1 | Tail 보정 후 재현율 |
|---|---:|---:|---:|---:|
| 고정 보정 last-2 dual | **90.08** | 89.02 | 원본 참조 | 원본 참조 |
| LLA soft | 88.60 | 88.90 | 88.99 | 90.00 |
| LLA hard 기준 | 89.16 | **89.36** | **89.47** | 90.47 |
| LLA hard 교환 OFF | 88.42 | 87.84 | 87.93 | 원본 참조 |
| self CE×confidence | 88.94 | 89.06 | 89.15 | 90.53 |
| self threshold 하한.8 | 88.26 | 88.56 | 88.67 | 89.33 |

교환 ON/OFF Major 정확도86.80 vs79.12. OFF는 self 억제도 복원했으므로 교환 정책 전체의 효과입니다.
Confidence 가중치와 하한.8 실험은 개선되지 않아 기존 하한.5·가중치1 유지.
고정 보정 last-2의 raw90.08이 LLA hard의 보정 후89.36보다 높아,
LLA가 모든 고정 보정보다 우수하다고 주장할 수 없습니다.
추가 seed2 기록은 전체 카탈로그에 있습니다.

## D. 이전 validation 학습 모델의 별도 official test 평가

**재학습이 아니라**, 위 validation 학습의 final50 체크포인트를 고정하고 공식 test를 평가했습니다.
최근 전체 train 새 학습(A)과 다릅니다.

| 출력 | seed1 | seed2 | 평균 |
|---|---:|---:|---:|
| Minor raw | 88.88 | 89.11 | 89.00 |
| Minor LLA | 89.40 | 89.27 | 89.34 |
| ensemble LLA | 89.02 | 88.93 | 88.98 |

## E. 과거 DeCon 결과의 정정

기존 [official-test-alpha12](../official-test-alpha12/README.md)의 DeCon 평균83.60은
초기 **불균형 unlabeled** 구성 결과입니다. 이후 Dual-matched 균형 unlabeled 재실험은
CIFAR-10 seed1/2/3 **91.11/90.71/90.92**, 평균**90.91**입니다.
당시 new dual ensemble 평균88.81보다 높습니다. 두 DeCon 결과를 섞으면 안 됩니다.
이번 새 학습(A)의 DeCon90.72와도 별도 실행으로 보관합니다.

## F. Tail 진단과 현재 해석

Validation 전용 체크포인트를 사용했습니다. 전체 train 학습 모델로 같은 이미지를
validation이라고 분석하지 않았습니다. Clean 진단과 weak augmentation5회 진단을 구분합니다.

- Online 가중치·두 seed·5회 weak 관측에서 실제 Tail self 선택94.85%.
- 오답 self843회 중 말619회. 보정이 정답→오답으로 만든 것은1회.
- 오답 self에서 Major도 같은 오답473회, Major만 정답118회.
- Major만 맞힌118회 모두 Tail 예측이므로 Head-only Major→Minor 규칙에서 제외.
- 오답 self526/843회는 confidence<.9였지만 같은 구간의 정답1093회도 존재.
- 신뢰도만 줄이거나 threshold를 높이는 후속 실험(C)은 이번 seed에서 개선되지 않았습니다.

이것은 고정 BN·eval-mode 모델의 진단이며, 실제 학습 중 오류 강화의 인과 증거는 아닙니다.
원본 `summary.json`, `failure_evidence.json`, 샘플별 CSV와 분석 Markdown을 아카이브했습니다.

## 재현 및 범위

실행·진단·아카이브 스크립트는 저장소 [scripts](../../scripts)에 있습니다.
Windows 로컬 경로가 포함된 스크립트는 환경에 맞게 경로를 변경해야 합니다.
원본 로그의 학습 경로 메시지는 전체 CIFAR archive를 읽는 것을 뜻할 수 있습니다.
축소 실험의 실제5000/400 선택은 `actual_data_audit.json`과 공통 인덱스로 검증했습니다.
샘플별 데이터는 클래스 정답을 진단에만 사용했으며, test로 가중치를 업데이트하지 않았습니다.
찾을 수 없는 과거 결과나 가중치만 남은 실험은 수치를 만들어 보완하지 않았습니다.
