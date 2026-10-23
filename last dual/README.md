# last dual

Dual 모델의 **검증셋 불일치 진단 기능 추가본** (2026-10-23).

## 현재 상태

- 기존 weighted CE, balanced sampler, Minor logit 보정 및 교환 학습 규칙 유지.
- 새로운 신뢰도 선택 규칙, 자기 의사 라벨 손실 차단, 보류 정책은 아직 적용하지 않음.
- 실제 검증 데이터 진단 결과와 새 학습 체크포인트는 이 폴더에 포함하지 않음.
- 합성 데이터로 예측 유형, 정오답 집계, 교환 마스크, warmup, 출력 및 test 차단 검증 완료.

## 기록 내용

각 검증 이미지의 Major 예측, Minor 보정 전 예측, Minor 보정 후 예측과 confidence를 기록한다.
보정으로 생성/해소된 불일치, 모델 자체 불일치, 세 예측 모두 다른 경우를 구분한다.
조건별 Major만 정답, Minor만 정답, 둘 다 오답, 보정의 오답→정답 및 정답→오답 횟수와 비율을 기록한다.
Head/Middle/Tail 예측 조합 및 기존 교환 후보·충돌·전달 마스크를 함께 기록한다.

## 실행

이 폴더로 이동하고 기존 실험에 사용한 PyTorch 환경을 사용한다.
필요 패키지: torch, torchvision, numpy, tensorboard, tqdm, Pillow, scikit-learn.

기존 학습 명령의 하이퍼파라미터를 유지하고 다음 옵션을 추가한다:

```text
--val-samples-per-class 500 --diagnose-disagreement --out results/last_dual_validation_seed1
```

실행 파일: `train_head_tail_exchange_sampler_alpha_val_split.py`.
`--val-samples-per-class 0`과 진단 옵션을 함께 주면 실행을 거부한다.
공식 test 학습 체크포인트를 validation 진단에 재사용하지 말 것: 해당 이미지가 학습에 포함되었을 수 있다.
새 검증 분할로 처음부터 학습하거나, 동일한 held-out 분할로 학습한 체크포인트만 사용한다.
`--eval-only` 경로에는 진단이 연결되어 있지 않다.
학습 중 평가 시점마다 기록하며, 최종 평가만 켜면 최종 평가 시점에 기록한다.

## 출력

`<out>/validation_disagreement/epoch_NNN/`

- `samples.csv`: 검증 이미지별 예측·confidence·정오답·교환 마스크.
- `conditions.csv`: 조건별 표본 수와 지표. 각 rate의 분모는 해당 행의 n이며 값은 0–1.
- `metadata.json`: seed, epoch, 보정 강도, threshold, 클래스 그룹 및 분석 제한.

validation_index는 해당 seed의 결정적 검증 분할 내 순서이다.
이는 clean validation / eval mode 분석이며 실제 학습의 augmented batch 기록은 아니다.
EMA 사용 시 EMA 모델로 분석한다.
서로 다른 모델의 confidence 및 보정 전후 confidence 차이는 보정된 신뢰도 점수가 아니며 승자 선택에 직접 사용하지 않는다.
조건별 표본 수와 seed 간 일관성을 확인한 뒤 규칙을 정하고, 규칙 선정과 성능 검증에는 별도 validation 분할 또는 교차검증을 사용한다.
최종 test는 규칙 확정 이후에만 사용한다.

## 파일 구성

원본 Dual 학습 스크립트와 진단 모듈, 실행에 필요한 dataset/models/utils 소스를 별도로 보관한다.
기존 저장소의 dual 폴더와 기존 실험 결과는 변경하지 않는다.
