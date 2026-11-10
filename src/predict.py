"""예측: 5-fold 교차 예측(out-of-fold)으로 고객 5,000명 전원의 이탈 확률을 낸다.

각 고객의 확률은 그 고객을 학습에 쓰지 않은 모델이 낸 값이다 → 처음 보는 고객을 예측하는 것과 같은 조건.
피처는 model.py 의 최종 피처셋(avg_watch_time_per_day 제외)을 쓴다.
실행: python3 src/predict.py
출력: reports/predictions.csv, reports/figures/roc_confusion.png
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict

from data import load
from model import CAT, NUM_ALL, SEED, TARGET, make_model
from risk_score import THRESHOLD, score
from segment import ORDER, assign

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "predictions.csv"
FIG = ROOT / "reports" / "figures"
NUM = [c for c in NUM_ALL if c != "avg_watch_time_per_day"]
CUT = 0.5  # 이 확률 이상이면 이탈로 예측


def oof_proba(df: pd.DataFrame, kind: str) -> np.ndarray:
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    return cross_val_predict(make_model(kind, NUM), df[NUM + CAT], df[TARGET],
                             cv=cv, method="predict_proba")[:, 1]


def plot(y: pd.Series, proba: dict, pred: pd.Series) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.5))
    for k, p in proba.items():
        fpr, tpr, _ = roc_curve(y, p)
        a1.plot(fpr, tpr, label=f"{k} (AUC {roc_auc_score(y, p):.3f})")
    a1.plot([0, 1], [0, 1], color="gray", ls="--", lw=1)
    a1.set_xlabel("false positive rate"); a1.set_ylabel("true positive rate")
    a1.set_title("ROC (out-of-fold)"); a1.legend()
    cm = confusion_matrix(y, pred)
    a2.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        a2.text(j, i, v, ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
    a2.set_xticks([0, 1], ["pred stay", "pred churn"]); a2.set_yticks([0, 1], ["actual stay", "actual churn"])
    a2.set_title(f"Confusion matrix (gbm, cut {CUT})")
    fig.tight_layout(); fig.savefig(FIG / "roc_confusion.png", dpi=120); plt.close(fig)


def main() -> None:
    df = load()
    y = df[TARGET]
    proba = {k: oof_proba(df, k) for k in ["logit", "gbm"]}
    for k, p in proba.items():
        pred = (p >= CUT).astype(int)
        tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
        print(f"{k:5s} OOF AUC={roc_auc_score(y, p):.4f} 정확도={(pred == y).mean():.4f} "
              f"정밀도={precision_score(y, pred):.3f} 재현율={recall_score(y, pred):.3f} "
              f"TN={tn} FP={fp} FN={fn} TP={tp}")

    p = proba["gbm"]
    out = pd.DataFrame({
        "customer_id": df.customer_id, "churned": y,
        "churn_proba": p.round(4), "churn_pred": (p >= CUT).astype(int),
        "churn_proba_logit": proba["logit"].round(4),
        "risk_score": score(df), "segment": assign(df)})
    out.to_csv(OUT, index=False)
    print("저장:", OUT, out.shape)

    tiers = pd.cut(out.churn_proba, [0, 0.2, 0.5, 0.8, 1], include_lowest=True)
    print("\n예측 확률 구간별 실제 이탈률\n",
          out.groupby(tiers, observed=True).churned.agg(["mean", "size"]).round(3).to_string())
    rule = (out.risk_score >= THRESHOLD).astype(int)
    print(f"\n모델 예측 vs 규칙(점수>={THRESHOLD}) 일치율: {(rule == out.churn_pred).mean():.4f}")
    seg = out.groupby("segment").agg(고객수=("churned", "size"), 평균예측확률=("churn_proba", "mean"),
                                     예측이탈률=("churn_pred", "mean"), 실제이탈률=("churned", "mean"))
    print("\n세그먼트별 예측\n", seg.reindex(ORDER).round(3).to_string())
    wrong = out[out.churn_pred != out.churned]
    print("\n틀린 예측", len(wrong), "건, 세그먼트별", wrong.segment.value_counts().to_dict())
    print("틀린 예측의 위험 점수 분포", wrong.risk_score.value_counts().sort_index().to_dict())
    plot(y, proba, out.churn_pred)


if __name__ == "__main__":
    main()
