"""라벨 생성 규칙 추정: 이진 플래그 점수합 (점수 >= 3 → 이탈).

가중치는 플래그 로지스틱 회귀 계수(약 12:12:-10:-6:6:5:3)를 정수로 단순화한 것.
실행: python3 src/risk_score.py
"""
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier, export_text

from data import load

WEIGHTS = {
    "watch_lt5": 2,      # 시청 5시간 미만
    "login_gt30": 2,     # 마지막 접속 30일 초과
    "watch_gt20": -2,    # 시청 20시간 초과
    "login_lt7": -1,     # 마지막 접속 7일 미만
    "basic": 1,          # Basic 요금제
    "crypto_gift": 1,    # Crypto·Gift Card 결제
    "profiles_le3": 1,   # 프로필 3개 이하
}
THRESHOLD = 3


def flags(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "watch_lt5": df.watch_hours < 5,
        "login_gt30": df.last_login_days > 30,
        "watch_gt20": df.watch_hours > 20,
        "login_lt7": df.last_login_days < 7,
        "basic": df.subscription_type == "Basic",
        "crypto_gift": df.payment_method.isin(["Crypto", "Gift Card"]),
        "profiles_le3": df.number_of_profiles <= 3,
    }).astype(int)


def score(df: pd.DataFrame) -> pd.Series:
    f = flags(df)
    return sum(f[k] * w for k, w in WEIGHTS.items())


def evidence(df: pd.DataFrame) -> None:
    """규칙을 세운 근거: 얕은 트리의 분기점, 플래그 로지스틱 계수."""
    cv = StratifiedKFold(5, shuffle=True, random_state=42)
    X = pd.get_dummies(df[["watch_hours", "last_login_days", "number_of_profiles",
                           "subscription_type", "payment_method"]], dtype=int)
    for d in [2, 3, 4, 5, 6, 8]:
        acc = cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=42), X, df.churned, cv=cv)
        print(f"트리 깊이 {d}: CV 정확도 {acc.mean():.4f}")
    tree = DecisionTreeClassifier(max_depth=3, random_state=42).fit(X, df.churned)
    print(export_text(tree, feature_names=list(X.columns)))

    f = flags(df)
    m = LogisticRegression(C=100, max_iter=2000)
    acc = cross_val_score(m, f, df.churned, cv=cv).mean()
    auc = cross_val_score(m, f, df.churned, cv=cv, scoring="roc_auc").mean()
    print(f"플래그 7개 로지스틱: CV 정확도 {acc:.4f}, AUC {auc:.4f}")
    m.fit(f, df.churned)
    print("계수", pd.Series(m.coef_[0], f.columns).round(2).to_dict(), "절편", round(m.intercept_[0], 2))


if __name__ == "__main__":
    df = load()
    evidence(df)
    pts = score(df)
    print(df.assign(pts=pts).groupby("pts").churned.agg(["mean", "size"]).round(3).T.to_string())
    print(f"규칙(점수>={THRESHOLD}) 정확도: {((pts >= THRESHOLD) == df.churned).mean():.4f}")
