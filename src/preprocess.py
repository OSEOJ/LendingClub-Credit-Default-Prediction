import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


TRAIN_FEATURES = [
    'id', 'hardship_flag', 'debt_settlement_flag', 'loan_status',
    'loan_amnt', 'int_rate', 'installment', 'grade', 'annual_inc', 'dti', 'delinq_2yrs',
    'inq_last_6mths', 'mths_since_last_delinq', 'mths_since_last_record', 'open_acc', 'revol_util', 'total_acc',
    'last_fico_range_high', 'collections_12_mths_ex_med', 'mths_since_last_major_derog',
    'tot_coll_amt', 'mths_since_rcnt_il', 'total_rev_hi_lim', 'acc_open_past_24mths',
    'avg_cur_bal', 'bc_open_to_buy', 'bc_util', 'mo_sin_old_il_acct', 'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mort_acc', 'mths_since_recent_bc',
    'mths_since_recent_bc_dlq', 'mths_since_recent_inq', 'mths_since_recent_revol_delinq',
    'num_accts_ever_120_pd', 'num_actv_bc_tl', 'num_actv_rev_tl', 'num_bc_sats',
    'num_il_tl', 'num_op_rev_tl', 'num_tl_op_past_12m', 'pct_tl_nvr_dlq', 'percent_bc_gt_75',
    'pub_rec_bankruptcies', 'tax_liens', 'tot_hi_cred_lim', 'total_bal_ex_mort',
    'total_bc_limit', 'total_il_high_credit_limit', 'home_ownership', 'verification_status',
    'purpose', 'title', 'addr_state', 'initial_list_status',
]

# 시간 기반 연체 변수 (결측치 = 최근 이벤트 없음으로 구간화)
TIME_FEATURES = [
    'mths_since_last_delinq', 'mths_since_recent_revol_delinq', 'collections_12_mths_ex_med',
    'mths_since_last_major_derog', 'mths_since_last_record', 'mths_since_recent_inq',
    'mo_sin_old_il_acct', 'mths_since_rcnt_il', 'mo_sin_old_rev_tl_op',
    'mo_sin_rcnt_rev_tl_op', 'mo_sin_rcnt_tl', 'mths_since_recent_bc', 'mths_since_recent_bc_dlq',
]

ONE_HOT_COLS = [
    'home_ownership', 'verification_status', 'purpose', 'title', 'addr_state', 'initial_list_status',
]

LOG_COLS = ['dti', 'revol_util', 'annual_inc']
BINARY_COLS = ['hardship_flag', 'debt_settlement_flag']
GRADE_MAP = {'A': 0, 'B': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6}


def load_data(filepath):
    data = pd.read_csv(filepath)
    cols_to_drop = [col for col in data.columns if col not in TRAIN_FEATURES]
    data.drop(columns=cols_to_drop, inplace=True)
    return data


def set_target(data):
    """loan_status를 이진 분류 타겟으로 변환. 불명확한 상태(Current 등)는 제거."""
    data['loan_status_binary'] = data['loan_status'].apply(
        lambda x: 1 if x == 'Fully Paid' else 0 if x in ['Charged Off', 'Default'] else None
    )
    data = data.dropna(subset=['loan_status_binary'])
    data.drop(columns=['loan_status'], inplace=True)
    return data


def handle_time_features(data, train_means=None):
    """
    시간 기반 변수를 3구간으로 인코딩.
      0 = 최근 이벤트 없음 (NaN)
      1 = 오래됨 (mean 초과)
      2 = 최근 (0 < x <= mean)

    train_means=None : 현재 데이터에서 구간 기준 계산 (학습 모드)
    train_means=dict : 학습 시 계산한 기준값 사용 (예측 모드, 데이터 누수 방지)
    """
    means = {}
    for col in TIME_FEATURES:
        if col in data.columns:
            mean_value = train_means[col] if train_means is not None else data[col][data[col] > 0].mean()
            means[col] = mean_value
            data[col] = data[col].apply(
                lambda x: 1 if pd.notna(x) and x > mean_value else
                          2 if pd.notna(x) and x > 0 else 0
            )
    data.dropna(inplace=True)
    return data, means


def scale_features(data):
    """퍼센트 변환 및 왜도 개선을 위한 log1p 변환."""
    for col in ['revol_util', 'int_rate']:
        if col in data.columns and data[col].dtype == object:
            data[col] = data[col].str.replace('%', '').astype(float) / 100
    for col in LOG_COLS:
        if col in data.columns:
            data[col] = np.log1p(data[col].clip(lower=0))  # 음수 클리핑 후 변환
    return data


def encode_features(data, use_onehot=True, fitted_ohe=None):
    """레이블/원핫 인코딩 및 이진 변수 변환.

    fitted_ohe=None  : OHE를 현재 데이터로 fit (학습 모드)
    fitted_ohe=object: 학습 시 저장된 OHE 사용 (예측 모드, 컬럼 일관성 보장)
    """
    data['id'] = pd.to_numeric(data['id'], errors='coerce').astype('Int64')

    # Y/N → 1/0 (NaN은 1로 처리)
    for col in BINARY_COLS:
        if col in data.columns:
            data[col] = data[col].map({'Y': 1, 'N': 0}).fillna(1).astype(int)

    # 순서형: grade — 고정 매핑으로 train/test 간 일관성 보장
    data['grade'] = data['grade'].map(GRADE_MAP)

    # 명목형 카테고리
    ohe = None
    if use_onehot:
        if fitted_ohe is None:
            ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
            ohe_data = ohe.fit_transform(data[ONE_HOT_COLS])
        else:
            ohe = fitted_ohe
            ohe_data = ohe.transform(data[ONE_HOT_COLS])
        ohe_df = pd.DataFrame.sparse.from_spmatrix(ohe_data, index=data.index)
        ohe_df.columns = ohe.get_feature_names_out(ONE_HOT_COLS)
        data = pd.concat([data.drop(columns=ONE_HOT_COLS), ohe_df], axis=1)
    else:
        data.drop(columns=ONE_HOT_COLS, inplace=True)

    return data, ohe


def preprocess(filepath, sample_frac=None, use_onehot=True, train_means=None, fitted_ohe=None):
    """
    전처리 파이프라인.

    반환: (data, artifacts)
      artifacts = {'means': dict, 'ohe': fitted OHE or None}

    학습 시: train_means=None, fitted_ohe=None → 내부에서 계산 후 artifacts에 포함
    예측 시: joblib.load()로 불러온 artifacts['means'], artifacts['ohe'] 전달
    """
    data = load_data(filepath)
    data = set_target(data)

    # 타겟 클래스 비율을 유지하는 층화 샘플링
    if sample_frac:
        data, _ = train_test_split(
            data, train_size=sample_frac,
            stratify=data['loan_status_binary'], random_state=42,
        )
        data = data.reset_index(drop=True)

    data, means = handle_time_features(data, train_means=train_means)
    data = scale_features(data)
    data, ohe = encode_features(data, use_onehot=use_onehot, fitted_ohe=fitted_ohe)

    return data, {'means': means, 'ohe': ohe}
