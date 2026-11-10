"""데이터 로딩 (입력 CSV 읽기는 이 모듈에서만)."""
from pathlib import Path

import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "data" / "netflix_customer_churn.csv"


def load() -> pd.DataFrame:
    return pd.read_csv(DATA)
