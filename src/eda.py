"""1단계 EDA: 데이터 품질 점검 + 변수별 이탈률 + 그림 저장.

실행: python3 src/eda.py  (프로젝트 루트에서)
출력: reports/figures/*.png, 표준출력 요약
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "netflix_customer_churn.csv"
FIG = ROOT / "reports" / "figures"

CATS = ["subscription_type", "payment_method", "number_of_profiles",
        "gender", "region", "device", "favorite_genre"]
NUMS = ["watch_hours", "last_login_days", "avg_watch_time_per_day", "age"]


def load() -> pd.DataFrame:
    return pd.read_csv(DATA)


def quality(df: pd.DataFrame) -> None:
    print("shape", df.shape)
    print("결측", int(df.isna().sum().sum()), "/ 중복행", int(df.duplicated().sum()),
          "/ 중복 ID", int(df.customer_id.duplicated().sum()))
    print("이탈률", round(df.churned.mean(), 3))
    # 요금제와 요금은 1:1 → 중복 정보
    print("요금제별 요금 종류 수", df.groupby("subscription_type").monthly_fee.nunique().to_dict())
    # avg_watch_time_per_day ≈ watch_hours / (last_login_days + 1) → 파생 변수
    derived = df.watch_hours / (df.last_login_days + 1)
    print("avg_watch 파생식 최대 오차", round((derived - df.avg_watch_time_per_day).abs().max(), 3))


def churn_by_category(df: pd.DataFrame) -> None:
    for c in CATS:
        t = pd.crosstab(df[c], df.churned)
        chi2, p, _, _ = stats.chi2_contingency(t)
        v = np.sqrt(chi2 / (len(df) * (min(t.shape) - 1)))
        print(f"{c:20s} p={p:.1e} CramerV={v:.3f}",
              df.groupby(c).churned.mean().round(3).to_dict())


def churn_by_numeric(df: pd.DataFrame) -> None:
    for c in NUMS:
        stay, churn = df[df.churned == 0][c], df[df.churned == 1][c]
        u = stats.mannwhitneyu(stay, churn)
        auc = u.statistic / (len(stay) * len(churn))
        print(f"{c:22s} 중앙값 유지={stay.median():.2f} 이탈={churn.median():.2f} "
              f"단변수AUC={max(auc, 1 - auc):.3f}")


def plot(df: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for ax, c in zip(axes.flat, CATS):
        rate = df.groupby(c).churned.mean().sort_values()
        rate.plot.barh(ax=ax, color="#4c78a8")
        ax.axvline(df.churned.mean(), color="gray", ls="--", lw=1)
        ax.set_title(c); ax.set_xlabel("churn rate"); ax.set_xlim(0, 1)
    axes.flat[-1].axis("off")
    fig.tight_layout(); fig.savefig(FIG / "churn_by_category.png", dpi=120); plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    for ax, c in zip(axes, NUMS):
        x = np.log1p(df[c]) if c == "avg_watch_time_per_day" else df[c]
        for k, color in [(0, "#4c78a8"), (1, "#e45756")]:
            ax.hist(x[df.churned == k], bins=40, alpha=0.6, color=color,
                    label="stay" if k == 0 else "churn")
        ax.set_title(c + (" (log1p)" if c == "avg_watch_time_per_day" else ""))
        ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "numeric_dist.png", dpi=120); plt.close(fig)

    # 구간별 이탈률 (임계값 확인용)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, c, step in [(axes[0], "last_login_days", 5), (axes[1], "watch_hours", 2)]:
        hi = 60 if c == "last_login_days" else 40
        bins = pd.cut(df[c].clip(upper=hi), np.arange(-0.01, hi + step, step))
        df.groupby(bins, observed=True).churned.mean().plot(ax=ax, marker="o")
        ax.set_title(f"churn rate by {c}"); ax.set_ylim(0, 1); ax.tick_params(axis="x", rotation=60)
    fig.tight_layout(); fig.savefig(FIG / "threshold_check.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    df = load()
    quality(df)
    churn_by_category(df)
    churn_by_numeric(df)
    plot(df)
    print("그림 저장:", FIG)
