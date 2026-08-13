# 머신러닝을 이용한 예측 분석 (Kaggle)

부트캠프 파이널 프로젝트. **Kaggle 대회 2개(분류·회귀)에 참가해 리더보드 상위 10% 진입**을 목표로, 전처리·EDA·피처엔지니어링·모델링·앙상블 전 과정을 수행.

- **분야:** 응용 ML · 정형데이터 · 부스팅 앙상블 · 하이퍼파라미터 최적화
- **유형:** 부트캠프 파이널 (8조, 5인)
- **내 역할:** **분류(흡연 예측) 파트**의 전처리·EDA·모델링(팀원 2인과 협업) + 발표

## 대회 1 — 분류: 생체신호를 이용한 흡연자 상태 이진 예측
- **평가지표:** ROC-AUC
- 데이터: 생체신호 22개 피처 + 타깃 `smoking`(0 비흡연 / 1 흡연). train (159,256 × 24). 클래스 분포 비흡연 57.6% / 흡연 42.4%.
- EDA: 타깃 분포, 변수 상관관계(키·몸무게·허리둘레 강상관, 혈압·콜레스테롤, 간수치 AST/ALT), 분포·이상치(IQR).
- 피처엔지니어링: 이상치가 몰린 피처(Gtp, HDL, LDL, ALT, AST, serum creatinine)에 `clip()`으로 임계값 조정(300/150/200/150/100/3).

## 대회 2 — 회귀: 전복(Abalone) 나이(Rings) 예측
- **평가지표:** RMSLE
- 데이터: 원본 train + 추가 abalone 데이터 병합(90,615 → 94,792행). 성별(Sex) OneHotEncoding, 타깃 로그변환 `np.log1p`.
- 파생변수: 표면적·밀도·BMI·각종 무게/길이 비율(ratio)·water_loss 등 다수 생성.

## 모델링 · 최적화
- 모델: **LightGBM · XGBoost · CatBoost** + 앙상블(Blending, Voting)
- 교차검증(fold 수 조정), 하이퍼파라미터 튜닝(GridSearchCV, **Optuna**).
- 피처 중요도: 무게 관련 피처(shell/shucked/whole/viscera weight)가 상위.

## 결과
- 회귀(전복): Blending(top3: cat·lgbm·rf, 10-fold) → 제출 RMSLE 0.14700(상위 25%) → **Voting + fold 조정(11/13/14) + 가중 앙상블**로 개선 → 최종 **RMSLE 0.14558**, **목표였던 리더보드 상위 10%(213등) 달성.**

## 기술 스택
`Python` `pandas` `scikit-learn` `LightGBM` `XGBoost` `CatBoost` `Optuna`

## 자료
- [`발표자료.pdf`](발표자료.pdf) — 전처리·EDA·모델·결과 전체.

> 본 저장소는 발표자료 중심의 문서형 정리입니다(대회 제출 노트북 코드는 별도 보관되어 있지 않아 미포함).
