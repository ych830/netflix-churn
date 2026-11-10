"""3단계 모델링: 로지스틱 vs 그래디언트 부스팅, 피처셋 비교, 중요도.

실행: python3 src/model.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from data import load

FIG = Path(__file__).resolve().parent.parent / "reports" / "figures"
SEED = 42
TARGET = "churned"
DROP = ["customer_id", "monthly_fee"]  # 식별자, 요금제와 1:1 중복
CAT = ["gender", "subscription_type", "region", "device", "payment_method", "favorite_genre"]
NUM_ALL = ["age", "watch_hours", "last_login_days", "number_of_profiles", "avg_watch_time_per_day"]


def make_model(kind: str, num: list[str]) -> Pipeline:
    if kind == "logit":
        pre = ColumnTransformer([("num", StandardScaler(), num),
                                 ("cat", OneHotEncoder(drop="first"), CAT)])
        clf = LogisticRegression(max_iter=1000)
    else:
        pre = ColumnTransformer([("num", "passthrough", num),
                                 ("cat", OrdinalEncoder(), CAT)])
        clf = HistGradientBoostingClassifier(
            categorical_features=list(range(len(num), len(num) + len(CAT))),
            random_state=SEED)
    return Pipeline([("pre", pre), ("clf", clf)])


def main() -> None:
    df = load().drop(columns=DROP)
    X, y = df.drop(columns=TARGET), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)

    feature_sets = {"all": NUM_ALL,
                    "no_avg_watch": [c for c in NUM_ALL if c != "avg_watch_time_per_day"]}
    print("== 5-fold CV ROC-AUC (train 80%) ==")
    for fs, num in feature_sets.items():
        for kind in ["logit", "gbm"]:
            s = cross_val_score(make_model(kind, num), X_tr[num + CAT], y_tr, cv=cv, scoring="roc_auc")
            print(f"{fs:13s} {kind:5s} {s.mean():.4f} ± {s.std():.4f}")

    # 최종: 파생 변수를 뺀 버전(원천 변수만으로 해석 가능) 으로 홀드아웃 평가
    num = feature_sets["no_avg_watch"]
    for kind in ["logit", "gbm"]:
        m = make_model(kind, num).fit(X_tr[num + CAT], y_tr)
        p = m.predict_proba(X_te[num + CAT])[:, 1]
        print(f"\n== holdout {kind} (no_avg_watch) AUC={roc_auc_score(y_te, p):.4f} ==")
        print(classification_report(y_te, (p >= 0.5).astype(int), digits=3))

        if kind == "logit":
            names = m.named_steps["pre"].get_feature_names_out()
            coef = pd.Series(m.named_steps["clf"].coef_[0], index=names).sort_values()
            print("로지스틱 계수 (표준화 기준, +면 이탈↑)\n", coef.round(3).to_string())
        else:
            imp = permutation_importance(m, X_te[num + CAT], y_te, scoring="roc_auc",
                                         n_repeats=10, random_state=SEED)
            s = pd.Series(imp.importances_mean, index=num + CAT).sort_values()
            print("순열 중요도 (AUC 감소량)\n", s.round(4).to_string())
            FIG.mkdir(parents=True, exist_ok=True)
            ax = s.plot.barh(figsize=(7, 5), color="#4c78a8")
            ax.set_title("Permutation importance (GBM, holdout AUC drop)")
            plt.tight_layout(); plt.savefig(FIG / "importance.png", dpi=120); plt.close()


if __name__ == "__main__":
    main()
