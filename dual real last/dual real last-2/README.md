# dual real last-2: LLA + hard Minor self loss

원본 `dual real last`는 보존하고, 그 안에 별도로 만든 비교 버전입니다.
Minor 본체의 자기학습만 soft CE에서 hard-label CE로 바꿉니다.
정식 학습은 이 코드 변경 과정에서 시작하지 않습니다.

## 유일한 학습 목적 변경

- Minor weak-view logits에 LLA를 적용해 soft 확률 q를 계산합니다.
- **Minor CNN 자기학습**: argmax(q)를 hard target으로 사용합니다.
- **LLA 보정 파라미터 학습**: q 전체를 그대로 사용하는 클래스 평균 loss 유지.
- **모델 간 교환**: 기존 hard target 유지.
- 기존 self 선택 마스크, last-2 충돌 차단 및 차단 전 분모를 유지합니다.
- LLA core 파일은 부모 버전과 동일합니다. 클래스 평균 정규화,
  logits/targets detach, optimizer 설정, 초기화는 바꾸지 않습니다.
- Major, weighted supervised CE, balanced sampler, 데이터와 seed, 학습률,
  EMA, warmup, batch size, 교환 조건, 평가 방식은 유지합니다.

이번 버전은 논문의 soft 자기학습을 그대로 재현한 모델이 아니라,
LLA 보정 + hard 자기학습을 비교하기 위한 Dual 변형입니다.
EFCC는 포함하지 않습니다.

## 실행 및 비교

`run_seed1_e50.ps1`: CIFAR-10, seed 1, 50 x 512 steps, batch 32, mu 7,
lr 0.03, sampler alpha 1.2, validation 500장/class, workers 2.
환경과 데이터 경로는 기존 로컬 설정을 사용합니다.
출력은 `dual_real_last_2_lla_hard_validation_cifar10_e50_s512_seed1`로 분리하며,
이미 존재하면 실행을 거부합니다. 동일 비교를 위해 처음부터 학습합니다.

체크포인트에는 `dual_real_last_2_minor_lla_hard` variant를 기록합니다.
soft 버전 체크포인트를 --resume으로 조용히 이어받지 못하게 차단합니다.
--init-checkpoint는 모델 가중치만 가져오는 기능으로,
LLA/optimizer를 포함한 동일 조건의 새 비교 실험과는 다릅니다.

## 보존되는 LLA 설정

- z_adjusted = z - log_softmax(lambda)
- lambda: 클래스별 파라미터, 0 초기화, Adam lr 0.002, weight decay 0
- epsilon 1e-8, 전체 K개 클래스 평균, unlabeled는 soft target mass로 정규화
- LLA loss는 lambda만 업데이트; CNN 자기학습은 lambda를 업데이트하지 않음
- LLA loss에는 모든 unlabeled 샘플 사용; CNN loss에는 기존 마스크 사용
- lambda EMA 없음; 평가 시 EMA CNN + 현재 lambda
- raw/adjusted Minor와 ensemble을 모두 기록
- best checkpoint는 비교를 위해 기존 raw ensemble 정확도로 선택
- 기존 sampler RNG의 bitwise resume까지 보장하는 구현은 아님

공식 저자 코드를 복제한 버전이 아닙니다. 초기화/optimizer/gradient 선택과
논문 전체 실험 설정의 차이는 부모 README에도 설명되어 있습니다.

## 검증

`python -B test_lla.py`
기존 LLA 테스트 8개에 다음 2개를 추가했습니다:
1. 실제 trainer의 self-loss 식이 hard CE이고 마스크/분모/gradient가 올바른지.
2. LLA 보정 loss에는 hard label이 아닌 soft probs_minor가 계속 입력되는지.

## 논문 근거

LLA core: Lee, Park, Kim, ICCV 2025, Sec. 4.2 Eqs. 9-15 및 Sec. 4.4.
https://openaccess.thecvf.com/content/ICCV2025/papers/Lee_Learnable_Logit_Adjustment_for_Imbalanced_Semi-Supervised_Learning_under_Class_Distribution_ICCV_2025_paper.pdf
