"""수익률 분석: IRR 계산, 무위험수익률 수집, Sharpe Ratio 산출."""
import datetime

import numpy as np
import numpy_financial as npf
import pandas as pd
from dateutil.relativedelta import relativedelta
from pandas_datareader import data as web


def calculate_actual_irr(row, n_months):
    """
    대출 상태에 따라 연환산 IRR을 계산합니다.

    Fully Paid:
        cash_flows = [-loan_amnt, installment × n_months]
    Charged Off / Default:
        지급 횟수 p ≈ total_pymnt / installment
        cash_flows = [-loan_amnt, installment × (p-1), installment + recoveries]
    """
    loan_amnt = row['loan_amnt']
    status = row['loan_status']
    installment = row['installment']
    recoveries = row.get('recoveries', 0) or 0

    if status == 'Fully Paid':
        cash_flows = [-loan_amnt] + [installment] * n_months

    elif status in ['Charged Off', 'Default']:
        total_pymnt = row['total_pymnt']
        p = max(int(round(total_pymnt / installment)) if installment else n_months // 2, 1)
        last_payment = installment + (recoveries if not pd.isna(recoveries) else 0)

        if p == 1:
            cash_flows = [-loan_amnt, last_payment]
        else:
            cash_flows = [-loan_amnt] + [installment] * (p - 1) + [last_payment]
    else:
        return None

    monthly_irr = npf.irr(cash_flows)
    if monthly_irr is None or not isinstance(monthly_irr, (int, float)):
        return None
    return (1 + monthly_irr) ** 12 - 1


def _fetch_monthly_rate(issue_d_list, ticker):
    """FRED에서 특정 ticker의 월별 평균 금리를 수집합니다."""
    results = []
    for val in issue_d_list:
        try:
            dt = datetime.datetime.strptime(val, '%b-%Y')
            start = dt.replace(day=1)
            end = (start + relativedelta(months=1)) - datetime.timedelta(days=1)
            df_rate = web.DataReader(ticker, 'fred', start, end)
            avg = df_rate[ticker].dropna().mean() / 100 if not df_rate.empty else np.nan
        except Exception:
            avg = np.nan
        results.append({'issue_d': val, 'risk_free_rate': avg})
    return pd.DataFrame(results)


def add_risk_free_rate(df_36, df_60):
    """
    36개월 대출 → DGS3 (미국채 3년물)
    60개월 대출 → DGS5 (미국채 5년물)
    """
    for df in (df_36, df_60):
        df['issue_d_str'] = df['issue_d'].apply(
            lambda x: x.strftime('%b-%Y') if isinstance(x, pd.Timestamp) else x
        )

    stats_3yr = _fetch_monthly_rate(df_36['issue_d_str'].dropna().unique(), 'DGS3')
    stats_5yr = _fetch_monthly_rate(df_60['issue_d_str'].dropna().unique(), 'DGS5')

    df_36['risk_free_rate'] = df_36['issue_d_str'].map(
        dict(zip(stats_3yr['issue_d'], stats_3yr['risk_free_rate']))
    )
    df_60['risk_free_rate'] = df_60['issue_d_str'].map(
        dict(zip(stats_5yr['issue_d'], stats_5yr['risk_free_rate']))
    )
    return df_36, df_60


def _weighted_std(values, weights):
    w_mean = np.sum(values * weights) / np.sum(weights)
    variance = np.sum(weights * (values - w_mean) ** 2) / np.sum(weights)
    return np.sqrt(variance)


def compute_portfolio_stats(df_all):
    """기존 LC 전략과 모델 전략의 수익률 및 Sharpe Ratio를 비교합니다."""
    df = df_all.dropna(subset=['actual_irr', 'risk_free_rate']).copy()
    w = df['loan_amnt']

    # 기존 전략
    df['excess_return'] = df['actual_irr'] - df['risk_free_rate']
    orig_irr = (w * df['actual_irr']).sum() / w.sum()
    orig_sharpe = (w * df['excess_return']).sum() / w.sum() / _weighted_std(df['excess_return'], w)

    # 모델 전략 (부도 예측 시 무위험자산 대체)
    df['model_return'] = df.apply(
        lambda row: row['actual_irr'] if row['predicted_y'] == 1 else row['risk_free_rate'],
        axis=1,
    )
    df['excess_return_model'] = df['model_return'] - df['risk_free_rate']
    model_irr = (w * df['model_return']).sum() / w.sum()
    model_sharpe = (w * df['excess_return_model']).sum() / w.sum() / _weighted_std(df['excess_return_model'], w)

    result = pd.DataFrame({
        'Strategy': ['Original LC', 'Our Model'],
        'Weighted Avg Return': [orig_irr, model_irr],
        'Sharpe Ratio': [orig_sharpe, model_sharpe],
    })
    print("\n--- 전략 비교 ---")
    print(result.to_string(index=False))
    return result


def run_analysis(predicted_csv_path):
    """전체 수익률 분석 파이프라인을 실행합니다."""
    df = pd.read_csv(predicted_csv_path)
    df = df[df['predicted_y'].notna()].copy()

    df['term'] = df['term'].str.strip()
    statuses = ['Fully Paid', 'Charged Off', 'Default']
    df_36 = df[df['term'] == '36 months'].drop(columns='term')
    df_60 = df[df['term'] == '60 months'].drop(columns='term')
    df_36 = df_36[df_36['loan_status'].isin(statuses)].copy()
    df_60 = df_60[df_60['loan_status'].isin(statuses)].copy()

    print("IRR 계산 중...")
    df_36['actual_irr'] = df_36.apply(lambda row: calculate_actual_irr(row, 36), axis=1)
    df_60['actual_irr'] = df_60.apply(lambda row: calculate_actual_irr(row, 60), axis=1)

    print("무위험수익률 수집 중 (FRED)...")
    df_36, df_60 = add_risk_free_rate(df_36, df_60)

    df_all = pd.concat([df_36, df_60], ignore_index=True)
    return compute_portfolio_stats(df_all)
