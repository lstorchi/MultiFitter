"""
Architecture search for the per-(v1, v2) transfer-learning models.

Pipeline evaluated for every candidate (same as the production script):
    pre-train on fitted data  ->  fine-tune on raw data  ->  score on raw data

Model selection uses grouped K-fold CV on (j1, j2) curves, so a curve is never
split between training and validation. A held-out test set of curves is kept
aside and only used once, for the final chosen model.
"""
import ast
import joblib
import numpy as np
import pandas as pd
import keras
from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------- settings
DATA_FILE = "modelling_data.npz"
V1, V2 = 9, 5
Y_MIN = 1e-2                 # rows with y below this are dropped
TEST_FRACTION = 0.2          # fraction of (j1, j2) curves held out for the final test
ES_FRACTION = 0.15           # fraction of training curves used only for early stopping
N_FOLDS = 4
SEEDS = [0, 1]               # repeat each fold with different seeds to measure noise
N_CANDIDATES = 10            # how many architectures to actually train
MIN_PARAMS = 300             # smallest model worth testing
MAX_PARAMS_PER_SAMPLE = 2.0  # budget: params <= this * n_fit_train_rows
ADD_BN_VARIANTS = False      # also test each candidate with BatchNorm after the first 2 layers
FREEZE_FIRST_N = 0           # Dense/BN layers to freeze during fine-tuning (0 = none)


# ---------------------------------------------------------------- data
def load_slice(path, v1, v2):
    d = np.load(path)
    out = {}
    for name in ("fit", "raw"):
        X, y = d[f"X{name}"], d[f"y{name}"]
        m = (X[:, 0] == v1) & (X[:, 1] == v2) & (y >= Y_MIN)
        out[name] = (X[m][:, 2:], np.log10(y[m]))   # features: j1, j2, es
    return out


def pair_groups(Xfit, Xraw):
    """Same integer id for the same (j1, j2) curve in both datasets."""
    pairs = np.unique(np.vstack([Xfit[:, :2], Xraw[:, :2]]), axis=0)
    lookup = {tuple(p): i for i, p in enumerate(pairs)}
    ids = lambda X: np.array([lookup[tuple(p)] for p in X[:, :2]])
    return ids(Xfit), ids(Xraw), len(pairs)


def split_groups(groups, fraction, seed):
    """Return (keep_ids, held_ids) of unique group ids."""
    uniq = np.unique(groups)
    gss = GroupShuffleSplit(n_splits=1, test_size=fraction, random_state=seed)
    keep, held = next(gss.split(uniq, groups=uniq))
    return uniq[keep], uniq[held]


# ---------------------------------------------------------------- candidates
def count_params(input_dim, shapes):
    n, prev = 0, input_dim
    for s in shapes:
        if s == "BN":
            n += 2 * prev                     # trainable gamma/beta
        else:
            n += prev * s + s
            prev = s
    return n + prev + 1                       # output layer


def candidate_pool():
    pool = []
    for depth in (2, 3, 4, 5):
        for w in (8, 16, 32, 64, 128, 256, 512):
            pool.append([w] * depth)                                  # constant width
            if w // 2 ** (depth - 1) >= 8:
                pool.append([w // 2 ** i for i in range(depth)])      # funnel (halving)
    unique = []
    for p in pool:
        if p not in unique:
            unique.append(p)
    return unique


def pick_candidates(input_dim, n_train_rows, k):
    """Keep architectures inside the parameter budget, then pick k spread on a log scale."""
    budget = max(MIN_PARAMS * 2, int(MAX_PARAMS_PER_SAMPLE * n_train_rows))
    pool = [(count_params(input_dim, s), s) for s in candidate_pool()]
    pool = sorted([p for p in pool if MIN_PARAMS <= p[0] <= budget], key=lambda p: p[0])
    if len(pool) <= k:
        chosen = [s for _, s in pool]
    else:
        logp = np.log10([p for p, _ in pool])
        targets = np.linspace(logp[0], logp[-1], k)
        idx = sorted({int(np.argmin(np.abs(logp - t))) for t in targets})
        chosen = [pool[i][1] for i in idx]
    if ADD_BN_VARIANTS:
        chosen += [[s[0], "BN", s[1], "BN", *s[2:]] if len(s) > 1 else [s[0], "BN"] for s in chosen]
    return chosen, budget


# ---------------------------------------------------------------- model
def build_model(input_dim, shapes):
    model = models.Sequential([layers.Input(shape=(input_dim,))])
    for s in shapes:
        model.add(layers.BatchNormalization() if s == "BN" else layers.Dense(s, activation="relu"))
    model.add(layers.Dense(1))
    return model


def early_stop():
    return EarlyStopping(monitor="val_loss", patience=10, min_delta=1e-6,
                         restore_best_weights=True, verbose=0)


def train_pipeline(shapes, fit_tr, fit_es, raw_tr, raw_es, seed):
    """Pre-train on fitted data, fine-tune on raw data. Each *_es set is only for early stopping."""
    keras.utils.set_random_seed(seed)
    model = build_model(fit_tr[0].shape[1], shapes)

    model.compile(optimizer=optimizers.Adam(1e-3), loss="mse")
    model.fit(*fit_tr, validation_data=fit_es, epochs=300, batch_size=256,
              callbacks=[early_stop()], verbose=0)

    for layer in model.layers[:FREEZE_FIRST_N]:
        layer.trainable = False
    model.compile(optimizer=optimizers.Adam(1e-4), loss="mse")
    model.fit(*raw_tr, validation_data=raw_es, epochs=300, batch_size=64,
              callbacks=[early_stop()], verbose=0)
    return model


def rmse_log10(model, X, y_true, scaler_y):
    pred = scaler_y.inverse_transform(model.predict(X, verbose=0).reshape(-1, 1)).ravel()
    return float(np.sqrt(np.mean((pred - y_true) ** 2)))


def prepare_fold(Xf, yf, gf, Xr, yr, gr, train_ids, eval_ids, seed):
    """Scale with one shared scaler, carve an early-stopping subset out of the training curves."""
    core_ids, es_ids = split_groups(train_ids, ES_FRACTION, seed)
    sel = lambda g, ids: np.isin(g, ids)

    scaler_X = StandardScaler().fit(np.vstack([Xf[sel(gf, train_ids)], Xr[sel(gr, train_ids)]]))
    scaler_y = StandardScaler().fit(np.concatenate([yf[sel(gf, train_ids)], yr[sel(gr, train_ids)]]).reshape(-1, 1))
    sx = scaler_X.transform
    sy = lambda y: scaler_y.transform(y.reshape(-1, 1)).ravel()

    part = lambda X, y, g, ids: (sx(X[sel(g, ids)]), sy(y[sel(g, ids)]))
    return dict(
        fit_tr=part(Xf, yf, gf, core_ids), fit_es=part(Xf, yf, gf, es_ids),
        raw_tr=part(Xr, yr, gr, core_ids), raw_es=part(Xr, yr, gr, es_ids),
        raw_eval=(sx(Xr[sel(gr, eval_ids)]), yr[sel(gr, eval_ids)]),   # y left in log10 units
        fit_eval=(sx(Xf[sel(gf, eval_ids)]), yf[sel(gf, eval_ids)]),
        scaler_X=scaler_X, scaler_y=scaler_y,
    )


# ---------------------------------------------------------------- main
if __name__ == "__main__":
    data = load_slice(DATA_FILE, V1, V2)
    Xf, yf = data["fit"]
    Xr, yr = data["raw"]
    gf, gr, n_pairs = pair_groups(Xf, Xr)

    dev_ids, test_ids = split_groups(np.arange(n_pairs), TEST_FRACTION, seed=42)
    n_fit_train = int(np.isin(gf, dev_ids).sum() * (1 - 1 / N_FOLDS) * (1 - ES_FRACTION))
    n_raw_train = int(np.isin(gr, dev_ids).sum() * (1 - 1 / N_FOLDS) * (1 - ES_FRACTION))

    candidates, budget = pick_candidates(Xf.shape[1], n_fit_train, N_CANDIDATES)

    print(f"\n=== v1={V1}, v2={V2} ===")
    print(f"Curves (j1,j2): {n_pairs}  ->  dev {len(dev_ids)}, test {len(test_ids)}")
    print(f"Rows: fit {len(yf)}, raw {len(yr)}  |  rows per curve: fit ~{len(yf)/n_pairs:.0f}, raw ~{len(yr)/n_pairs:.0f}")
    print(f"Approx. training rows per fold: fit {n_fit_train}, raw {n_raw_train}")
    print(f"Parameter budget: {MIN_PARAMS} .. {budget}")
    for s in candidates:
        print(f"  {str(s):40s} {count_params(Xf.shape[1], s):>8d} params")
    if len(dev_ids) < N_FOLDS * 2:
        raise SystemExit(f"Only {len(dev_ids)} dev curves: reduce N_FOLDS or TEST_FRACTION.")

    # ---- grouped CV
    rows = []
    gkf = GroupKFold(n_splits=N_FOLDS)
    for fold, (tr, ev) in enumerate(gkf.split(dev_ids, groups=dev_ids)):
        for seed in SEEDS:
            f = prepare_fold(Xf, yf, gf, Xr, yr, gr, dev_ids[tr], dev_ids[ev], seed)
            for shapes in candidates:
                model = train_pipeline(shapes, f["fit_tr"], f["fit_es"], f["raw_tr"], f["raw_es"], seed)
                rows.append(dict(
                    arch=str(shapes), params=count_params(Xf.shape[1], shapes), fold=fold, seed=seed,
                    raw_rmse=rmse_log10(model, *f["raw_eval"], f["scaler_y"]),
                    fit_rmse=rmse_log10(model, *f["fit_eval"], f["scaler_y"]),
                ))
                print(f"fold {fold} seed {seed} {rows[-1]['arch']:40s} raw RMSE {rows[-1]['raw_rmse']:.4f}")
                keras.backend.clear_session()

    res = pd.DataFrame(rows)
    res.to_csv(f"arch_search_runs_v{V1}_{V2}.csv", index=False)
    summary = (res.groupby(["arch", "params"])
                  .agg(raw_mean=("raw_rmse", "mean"), raw_std=("raw_rmse", "std"),
                       fit_mean=("fit_rmse", "mean"), n=("raw_rmse", "size"))
                  .reset_index().sort_values("raw_mean"))
    summary["raw_se"] = summary["raw_std"] / np.sqrt(summary["n"])
    summary.to_csv(f"arch_search_summary_v{V1}_{V2}.csv", index=False)

    # ---- one-standard-error rule: smallest model statistically tied with the best
    best = summary.iloc[0]
    tied = summary[summary["raw_mean"] <= best["raw_mean"] + best["raw_se"]]
    chosen = tied.sort_values("params").iloc[0]
    print("\n=== CV summary (log10 RMSE on raw data, lower is better) ===")
    print(summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nBest mean:  {best['arch']}  ({best['raw_mean']:.4f} ± {best['raw_se']:.4f})")
    print(f"Chosen (1-SE rule, smallest tied model): {chosen['arch']}  ({chosen['params']} params)")
    if best["params"] == summary["params"].max():
        print("WARNING: the best model is the LARGEST tested -> raise MAX_PARAMS_PER_SAMPLE and rerun.")
    if best["params"] == summary["params"].min():
        print("WARNING: the best model is the SMALLEST tested -> lower MIN_PARAMS and rerun.")

    # ---- final fit on all dev curves, single evaluation on the held-out test curves
    shapes = ast.literal_eval(chosen["arch"])
    f = prepare_fold(Xf, yf, gf, Xr, yr, gr, dev_ids, test_ids, seed=0)
    model = train_pipeline(shapes, f["fit_tr"], f["fit_es"], f["raw_tr"], f["raw_es"], seed=0)
    print(f"\nHeld-out test log10 RMSE: raw {rmse_log10(model, *f['raw_eval'], f['scaler_y']):.4f}, "
          f"fit {rmse_log10(model, *f['fit_eval'], f['scaler_y']):.4f}")
    model.save(f"model_v{V1}_{V2}.keras")
    joblib.dump({"scaler_X": f["scaler_X"], "scaler_y": f["scaler_y"]}, f"scalers_v{V1}_{V2}.joblib")
