"""
LendingClub 채무불이행 예측 및 수익률 최대화 - 학습 파이프라인

실행:
    python train.py
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess
from src.feature_engineer import run_feature_engineering
from src.model import train, plot_confusion_matrix, plot_shap
from src.analysis import run_analysis

DATA_PATH = './data/lending_club_2020_train.csv'
OUTPUT_DIR = './output'
SAMPLE_FRAC = 0.05
MODEL_PATH = f'{OUTPUT_DIR}/model'
ARTIFACTS_PATH = f'{OUTPUT_DIR}/preprocess_artifacts.pkl'
PREDICTIONS_PATH = f'{OUTPUT_DIR}/predictions.csv'

os.makedirs(OUTPUT_DIR, exist_ok=True)


# A. 전처리
print("=" * 50)
print("A. 전처리")
print("=" * 50)
data, artifacts = preprocess(DATA_PATH, sample_frac=SAMPLE_FRAC, use_onehot=True)
joblib.dump(artifacts, ARTIFACTS_PATH)  # predict.py에서 동일한 기준으로 전처리하기 위해 저장
print(f"전처리 완료: {data.shape}")
print(f"전처리 artifacts 저장 → {ARTIFACTS_PATH}")

# B. 피처 엔지니어링 (사후 정보 예측 모델)
print("\n" + "=" * 50)
print("B. 피처 엔지니어링 (hardship / debt_settlement 예측)")
print("=" * 50)
data = run_feature_engineering(data)
print(f"피처 엔지니어링 완료: {data.shape}")

# C. 채무불이행 예측 모델 학습
print("\n" + "=" * 50)
print("C. 채무불이행 예측 모델 학습 (GBC)")
print("=" * 50)
model, data, x_test = train(data, model_type='gbc', model_path=MODEL_PATH)
plot_confusion_matrix(model, output_path=f'{OUTPUT_DIR}/confusion_matrix.png')
plot_shap(model, x_test, output_path=f'{OUTPUT_DIR}/shap_summary.png')

# D. 예측 결과를 원본 데이터와 합쳐 저장 (IRR 분석에 필요한 원본 컬럼 보존)
print("\n" + "=" * 50)
print("D. 예측 결과 저장")
print("=" * 50)
data_orig = pd.read_csv(DATA_PATH)

# 전처리와 동일한 층화 샘플 추출
target_temp = data_orig['loan_status'].apply(
    lambda x: 1 if x == 'Fully Paid' else 0 if x in ['Charged Off', 'Default'] else None
)
valid_idx = target_temp.dropna().index
data_orig_valid = data_orig.loc[valid_idx]
data_orig_sample, _ = train_test_split(
    data_orig_valid, train_size=SAMPLE_FRAC,
    stratify=target_temp.loc[valid_idx], random_state=42,
)
data_orig_sample = data_orig_sample.reset_index(drop=True)

data_orig_sample['id'] = pd.to_numeric(data_orig_sample['id'], errors='coerce').astype('Int64')
data['id'] = pd.to_numeric(data['id'], errors='coerce').astype('Int64')
data.set_index('id', inplace=True)

mask = data_orig_sample['id'].isin(data.index)
data_orig_sample.loc[mask, 'predicted_y'] = data.loc[data_orig_sample.loc[mask, 'id'], 'predicted_y'].values
data_orig_sample.to_csv(PREDICTIONS_PATH, index=False)
print(f"예측 결과 저장 완료 → {PREDICTIONS_PATH}")

# E. 수익률 분석
print("\n" + "=" * 50)
print("E. 수익률 분석 (IRR / Sharpe Ratio)")
print("=" * 50)
run_analysis(PREDICTIONS_PATH)
