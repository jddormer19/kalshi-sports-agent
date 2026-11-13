"""Time-split training: LightGBM learns a correction on top of the market's log-odds."""
import json

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .collect import DATA
from .features import FEATURES, _logit

PARAMS = dict(objective="binary", learning_rate=0.02, num_leaves=15, min_data_in_leaf=200,
              feature_fraction=0.7, bagging_fraction=0.8, bagging_freq=1, lambda_l2=10.0,
              cat_smooth=50, verbose=-1, seed=0)


def split(df: pd.DataFrame, q1=0.6, q2=0.8):
    """train: markets closed before T1. val: snapshots at/after T1 on markets closed before T2.
    test: snapshots at/after T2. No training label is from a market that closed after val/test starts."""
    close_s = df.close_time.map(pd.Timestamp.timestamp)
    t1, t2 = np.quantile(close_s, [q1, q2])
    tr = df[close_s < t1]
    va = df[(df.t0 >= t1) & (close_s < t2)]
    te = df[df.t0 >= t2]
    return tr, va, te, (t1, t2)


def wbrier(y, p, w):
    return float(np.average((p - y) ** 2, weights=w))


def wlogloss(y, p, w):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return float(np.average(-(y * np.log(p) + (1 - y) * np.log(1 - p)), weights=w))


class Model:
    """Market log-odds + boosted correction, then 2-parameter Platt recalibration."""

    def __init__(self, booster, platt, categories=None):
        self.booster, self.platt, self.categories = booster, platt, categories

    def raw_logit(self, X):
        return _logit(X.mid.to_numpy()) + self.booster.predict(X[FEATURES], raw_score=True)

    def predict(self, X):
        return self.platt.predict_proba(self.raw_logit(X).reshape(-1, 1))[:, 1]


def fit(tr, va):
    ds = lambda d: lgb.Dataset(d[FEATURES], d.y, weight=d.w, init_score=_logit(d.mid.to_numpy()),
                               categorical_feature=["category"], free_raw_data=False)
    dtr, dva = ds(tr), ds(va)
    booster = lgb.train(PARAMS, dtr, num_boost_round=2000, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(100, verbose=False)])
    m = Model(booster, None, list(tr.category.cat.categories))
    m.platt = LogisticRegression(C=1e6).fit(m.raw_logit(va).reshape(-1, 1), va.y, sample_weight=va.w)
    return m


def report(te, preds: dict) -> pd.DataFrame:
    rows = []
    for name, p in preds.items():
        rows.append(dict(model=name, brier=wbrier(te.y, p, te.w), logloss=wlogloss(te.y, p, te.w)))
    out = pd.DataFrame(rows).set_index("model")
    out["brier_vs_market_%"] = 100 * (out.brier / out.loc["market_mid", "brier"] - 1)
    return out


def breakdown(te, p_model, by):
    g = te.assign(pm=p_model).groupby(by, observed=True)
    return g.apply(lambda d: pd.Series(dict(
        n=len(d), markets=d.ticker.nunique(), yes_rate=np.average(d.y, weights=d.w),
        brier_market=wbrier(d.y, d.mid, d.w), brier_model=wbrier(d.y, d.pm, d.w))), include_groups=False)


def reliability_plot(te, p_model, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 1, 11)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for name, p in [("market mid", te.mid.to_numpy()), ("model", p_model)]:
        idx = np.clip(np.digitize(p, bins) - 1, 0, 9)
        d = pd.DataFrame(dict(b=idx, p=p, y=te.y, w=te.w)).groupby("b").apply(
            lambda g: pd.Series(dict(p=np.average(g.p, weights=g.w), y=np.average(g.y, weights=g.w))),
            include_groups=False)
        ax.plot(d.p, d.y, "o-", label=name)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    ax.set(xlabel="predicted P(YES)", ylabel="observed YES rate", title="Test-set reliability")
    ax.legend(); fig.tight_layout(); fig.savefig(path, dpi=120)


def run(hours=None):
    df = pd.read_parquet(DATA / "features.parquet")
    df = df[df.y >= 0]
    tr, va, te, (t1, t2) = split(df)
    fmt = lambda t: pd.Timestamp(t, unit="s").date()
    print(f"train {len(tr)} snaps / {tr.ticker.nunique()} mkts (close < {fmt(t1)}) | "
          f"val {len(va)} / {va.ticker.nunique()} | test {len(te)} / {te.ticker.nunique()} (t0 >= {fmt(t2)})")

    model = fit(tr, va)
    # benchmark: market price recalibrated with a 2-param logistic (captures favourite-longshot bias)
    recal = LogisticRegression(C=1e6).fit(_logit(tr.mid.to_numpy()).reshape(-1, 1), tr.y, sample_weight=tr.w)
    p_model = model.predict(te)
    preds = {"market_mid": te.mid.to_numpy(),
             "market_recalibrated": recal.predict_proba(_logit(te.mid.to_numpy()).reshape(-1, 1))[:, 1],
             "model": p_model}
    rep = report(te, preds)
    print(f"\nTest metrics (weighted; boosting rounds = {model.booster.best_iteration}):\n{rep.round(5).to_string()}")

    te = te.assign(price_bucket=pd.cut(te.mid, [0, .05, .15, .35, .65, .85, .95, 1]))
    print("\nBy category:\n" + breakdown(te, p_model, "category").round(4).to_string())
    print("\nBy price bucket:\n" + breakdown(te, p_model, "price_bucket").round(4).to_string())

    imp = pd.Series(model.booster.feature_importance("gain"), index=FEATURES).sort_values(ascending=False)
    print("\nTop features (gain):\n" + (imp / imp.sum()).head(10).round(3).to_string())

    reliability_plot(te, p_model, DATA / "reliability.png")
    joblib.dump(model, DATA / "model.joblib")
    te.drop(columns="price_bucket").assign(p_model=p_model).to_parquet(DATA / "test_preds.parquet")
    (DATA / "metrics.json").write_text(json.dumps(rep.reset_index().to_dict("records"), indent=1))
    print(f"\nsaved model, test predictions and reliability plot to {DATA}")
