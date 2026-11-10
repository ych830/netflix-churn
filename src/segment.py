"""5단계 세그먼트: 위험 점수대 × 주요 원인, 레버별 what-if.

세그먼트(상호배타, 위에서부터 먼저 해당하는 것):
  안전(점수<=1) / 경계(2) / 고위험(>=3) 중
    - 이탈 확정형: 장기 미접속(>30일) AND 저시청(<5h)
    - 휴면형: 장기 미접속(>30일)
    - 저시청형: 저시청(<5h)
    - 조건형: 참여도는 보통인데 요금제·결제·프로필 조건이 쌓임

what-if 는 risk_score 규칙 위에서 플래그 1개를 바꿨을 때 점수가 3 미만이 되는지 계산한다.
인과 효과가 아니라 '규칙상 구제 가능한 인원'이다.
실행: python3 src/segment.py
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from data import load
from risk_score import THRESHOLD, WEIGHTS, flags, score

FIG = Path(__file__).resolve().parent.parent / "reports" / "figures"
ORDER = ["안전", "경계", "고위험-이탈확정형", "고위험-휴면형", "고위험-저시청형", "고위험-조건형"]

# 회사가 움직일 수 있는 레버: 플래그 → 레버 이름 (해당 플래그를 0으로 만드는 행동)
LEVERS = {
    "login_gt30": "재방문 유도 (30일 내 접속)",
    "watch_lt5": "시청 5시간 이상으로 끌어올림",
    "basic": "Basic → 상위 요금제 전환",
    "crypto_gift": "카드/PayPal 정기결제 전환",
    "profiles_le3": "프로필 4개 이상 (가족 공유)",
}


def assign(df: pd.DataFrame) -> pd.Series:
    pts, f = score(df), flags(df)
    seg = np.select(
        [pts <= 1, pts == 2,
         (f.login_gt30 == 1) & (f.watch_lt5 == 1),
         f.login_gt30 == 1,
         f.watch_lt5 == 1],
        ORDER[:5], default=ORDER[5])
    return pd.Series(seg, index=df.index, name="segment")


def summary(df: pd.DataFrame) -> pd.DataFrame:
    d = df.assign(segment=assign(df), pts=score(df))
    g = d.groupby("segment").agg(
        고객수=("churned", "size"), 이탈률=("churned", "mean"), 평균점수=("pts", "mean"),
        시청h_중앙=("watch_hours", "median"), 미접속일_중앙=("last_login_days", "median"),
        Basic비율=("subscription_type", lambda s: (s == "Basic").mean()),
        CryptoGift비율=("payment_method", lambda s: s.isin(["Crypto", "Gift Card"]).mean()),
        프로필3이하비율=("number_of_profiles", lambda s: (s <= 3).mean()))
    g = g.reindex(ORDER)
    g.insert(1, "비중", g.고객수 / len(d))
    g["이탈자수"] = (g.고객수 * g.이탈률).round().astype(int)
    return g


def what_if(df: pd.DataFrame) -> pd.DataFrame:
    """고위험(점수>=3) 고객 중 레버 1개만으로 점수<3 이 되는 인원 (세그먼트별)."""
    pts, f, seg = score(df), flags(df), assign(df)
    hi = pts >= THRESHOLD
    rows = {}
    for flag, name in LEVERS.items():
        saved = hi & (f[flag] == 1) & (pts - WEIGHTS[flag] < THRESHOLD)
        rows[name] = saved.groupby(seg).sum().reindex(ORDER[2:], fill_value=0)
    t = pd.DataFrame(rows).T
    t["합계"] = t.sum(axis=1)
    return t


def plot(g: pd.DataFrame) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    labels = ["safe", "borderline", "hi:lost", "hi:dormant", "hi:low-watch", "hi:conditions"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(labels, g.고객수, color="#c7d3e3", label="customers")
    ax.bar(labels, g.이탈자수, color="#e45756", label="churned")
    for i, (n, r) in enumerate(zip(g.고객수, g.이탈률)):
        ax.text(i, n + 20, f"{r:.0%}", ha="center")
    ax.set_title("Segments: size and churn rate"); ax.legend()
    fig.tight_layout(); fig.savefig(FIG / "segments.png", dpi=120); plt.close(fig)


if __name__ == "__main__":
    df = load()
    g = summary(df)
    pd.set_option("display.width", 220)
    print(g.round(3).to_string())
    print("\n고위험 레버별 '1개만 바꾸면 구제' 인원 (규칙 기준)\n", what_if(df).to_string())
    assert g.고객수.sum() == len(df), "세그먼트가 전체 고객을 덮지 않음"
    plot(g)
