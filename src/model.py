"""채무불이행 예측 모델: PyCaret 기반 학습, 평가, 저장/로드."""
import numpy as np
import pandas as pd
import shap
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score, confusion_matrix
from pycaret.classification import (
    setup, create_model, predict_model, save_model,
    load_model as pycaret_load_model,
)


SCALE_NORMAL = ['annual_inc', 'loan_amnt', 'int_rate', 'dti', 'installment']
SCALE_BIMODAL = ['revol_util']


def _find_best_threshold(y_true, y_prob):
    """F1-score 기준 최적 임계값 탐색."""
    best_threshold, best_score = 0.5, 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def train(data, model_type='gbc', model_path='output/model'):
    # id는 식별자이므로 모델 피처에서 제외
    X = data.drop(columns=['loan_status_binary', 'id'], errors='ignore')
    y = data['loan_status_binary']
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # x_test 스케일링은 SHAP 분석용 — PyCaret은 pycaret_data를 직접 받아 내부적으로 전처리함
    scaler_normal = StandardScaler()
    scaler_bimodal = RobustScaler()
    x_train[SCALE_NORMAL] = scaler_normal.fit_transform(x_train[SCALE_NORMAL])
    x_train[SCALE_BIMODAL] = scaler_bimodal.fit_transform(x_train[SCALE_BIMODAL])
    x_test[SCALE_NORMAL] = scaler_normal.transform(x_test[SCALE_NORMAL])
    x_test[SCALE_BIMODAL] = scaler_bimodal.transform(x_test[SCALE_BIMODAL])

    pycaret_data = data.drop(columns=['id'], errors='ignore')
    model = setup(
        data=pycaret_data, target='loan_status_binary', session_id=123,
        imputation_type='simple', numeric_imputation='mean', categorical_imputation='mode',
    )
    model = create_model(model_type)

    # 임계값 탐색은 x_test 인덱스 행만 사용 (훈련 데이터 누수 방지)
    pycaret_test = pycaret_data.loc[x_test.index]
    predictions_test = predict_model(model, data=pycaret_test, raw_score=True)
    best_threshold, best_score = _find_best_threshold(
        predictions_test['loan_status_binary'], predictions_test['prediction_score_1']
    )
    print(f"Best threshold: {best_threshold:.2f}, F1: {best_score:.4f}")

    # 전체 데이터에 predicted_y 배정
    predictions_all = predict_model(model, data=pycaret_data, raw_score=True)
    data['predicted_y'] = (predictions_all['prediction_score_1'] >= best_threshold).values

    save_model(model, model_path)
    print(f"Model saved → {model_path}")

    return model, data, x_test


def predict(model_path, data):
    model = pycaret_load_model(model_path)
    data_no_id = data.drop(columns=['id'], errors='ignore')
    predictions = predict_model(model, data=data_no_id)
    data['predicted_y'] = predictions['prediction_label'].values
    return model, data


def plot_confusion_matrix(model, output_path='output/confusion_matrix.png'):
    predictions = predict_model(model)
    y_true = predictions['loan_status_binary']
    y_pred = predictions['prediction_label']
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Oranges', linewidths=1, linecolor='black')
    plt.title("Confusion Matrix", fontsize=14, fontweight='bold')
    plt.xlabel("Predicted Label", fontsize=12)
    plt.ylabel("True Label", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"Confusion matrix saved → {output_path}")


def plot_shap(model, x_test, output_path='output/shap_summary.png'):
    explainer = shap.Explainer(model)
    shap_values = explainer.shap_values(x_test)
    shap.summary_plot(shap_values, x_test, show=False)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()
    print(f"SHAP summary saved → {output_path}")
