"""
LendingClub 채무불이행 예측 - 테스트 데이터 평가

사전에 train.py를 실행하여 output/model, output/preprocess_artifacts.pkl 이 저장되어 있어야 합니다.

실행:
    python predict.py
"""
import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split

from src.preprocess import preprocess
from src.feature_engineer import run_feature_engineering
from src.model import predict
from src.analysis import run_analysis

DATA_PATH = './data/lending_club_2020_test.csv'
OUTPUT_DIR = './output'
MODEL_PATH = f'{OUTPUT_DIR}/model'
ARTIFACTS_PATH = f'{OUTPUT_DIR}/preprocess_artifacts.pkl'
PREDICTIONS_PATH = f'{OUTPUT_DIR}/predictions_test.csv'
SAMPLE_FRAC = 0.05

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 학습 시 저장된 전처리 기준값(시간 피처 mean, OHE) 로드
artifacts = joblib.load(ARTIFACTS_PATH)

# A. 전처리 (학습과 동일한 기준값 적용 → 데이터 누수 방지, 컬럼 일관성 보장)
print("=" * 50)
print("A. 전처리")
print("=" * 50)
data, _ = preprocess(
    DATA_PATH, sample_frac=SAMPLE_FRAC, use_onehot=True,
    train_means=artifacts['means'], fitted_ohe=artifacts['ohe'],
)
print(f"전처리 완료: {data.shape}")

# B. 피처 엔지니어링
print("\n" + "=" * 50)
print("B. 피처 엔지니어링")
print("=" * 50)
data = run_feature_engineering(data)
print(f"피처 엔지니어링 완료: {data.shape}")

# C. 저장된 모델로 예측
print("\n" + "=" * 50)
print("C. 예측 (저장된 모델 로드)")
print("=" * 50)
model, data = predict(MODEL_PATH, data)
print(f"예측 완료: {data['predicted_y'].value_counts().to_dict()}")

# D. 예측 결과를 원본 데이터와 합쳐 저장
print("\n" + "=" * 50)
print("D. 예측 결과 저장")
print("=" * 50)
data_orig = pd.read_csv(DATA_PATH)

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
print("E. 수익률 분석")
print("=" * 50)
run_analysis(PREDICTIONS_PATH)
