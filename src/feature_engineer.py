"""사후 정보 예측 모델: hardship_flag 및 debt_settlement_flag를 사전 정보로 예측."""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import f1_score
from pycaret.classification import setup, create_model, predict_model


SCALE_NORMAL = ['annual_inc', 'loan_amnt', 'int_rate', 'dti', 'installment']
SCALE_BIMODAL = ['revol_util']


def _find_best_threshold(y_true, y_prob):
    """F1-score 기준 최적 임계값 탐색 (소수 클래스 탐지에 적합)."""
    best_threshold, best_score = 0.5, 0.0
    for threshold in np.linspace(0.0, 1.0, 101):
        score = f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        if score > best_score:
            best_score = score
            best_threshold = threshold
    return best_threshold, best_score


def _train_flag_model(data, target_col, drop_cols, model_type):
    X = data.drop(columns=drop_cols)
    y = data[target_col]

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)
    x_train = x_train.astype('float32')
    y_train = y_train.astype('float32')

    scaler_normal = StandardScaler()
    scaler_bimodal = RobustScaler()
    x_train[SCALE_NORMAL] = scaler_normal.fit_transform(x_train[SCALE_NORMAL])
    x_train[SCALE_BIMODAL] = scaler_bimodal.fit_transform(x_train[SCALE_BIMODAL])
    x_test[SCALE_NORMAL] = scaler_normal.transform(x_test[SCALE_NORMAL])
    x_test[SCALE_BIMODAL] = scaler_bimodal.transform(x_test[SCALE_BIMODAL])

    train_df = x_train.copy()
    train_df[target_col] = y_train.values

    model = setup(
        data=train_df, target=target_col, session_id=123,
        preprocess=False, fix_imbalance=True, normalize=False,
    )
    model = create_model(model_type)

    # 임계값 탐색은 x_test에서만 수행 (훈련 데이터 누수 방지)
    test_df = x_test.copy()
    test_df[target_col] = y_test.values
    predictions_test = predict_model(model, data=test_df, raw_score=True)
    best_threshold, best_score = _find_best_threshold(
        predictions_test[target_col], predictions_test['prediction_score_1']
    )
    print(f"[{target_col}] Best threshold: {best_threshold:.2f}, F1: {best_score:.4f}")

    # 전체 데이터에 대한 예측 스코어 추출 (인덱스 기반 정렬)
    full_X = pd.concat([x_train, x_test])
    full_X[target_col] = pd.concat([y_train, y_test]).values
    predictions_full = predict_model(model, data=full_X, raw_score=True)
    score_series = pd.Series(predictions_full['prediction_score_1'].values, index=full_X.index)

    return score_series, best_threshold


def predict_hardship_flag(data):
    drop_cols = ['hardship_flag', 'loan_status_binary', 'debt_settlement_flag']
    score_series, threshold = _train_flag_model(data, 'hardship_flag', drop_cols, 'et')

    data['hardship_flag_predict_score'] = score_series
    data['hardship_flag_predict'] = (data['hardship_flag_predict_score'] >= threshold).astype(int)
    data.drop(columns=['hardship_flag'], inplace=True)
    return data


def predict_debt_settlement_flag(data):
    drop_cols = [
        'debt_settlement_flag', 'loan_status_binary',
        'hardship_flag_predict_score', 'hardship_flag_predict',
    ]
    score_series, threshold = _train_flag_model(data, 'debt_settlement_flag', drop_cols, 'et')

    data['debt_settlement_flag_predict_score'] = score_series
    data['debt_settlement_flag_predict'] = (data['debt_settlement_flag_predict_score'] >= threshold).astype(int)
    data.drop(columns=['debt_settlement_flag'], inplace=True)
    return data


def run_feature_engineering(data):
    data = predict_hardship_flag(data)
    data = predict_debt_settlement_flag(data)
    return data
