import sys
import joblib
import pickle
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel as C, Matern, WhiteKernel

def curves_matrix(X, y, egrid):
    """One row per curve, one column per energy (X, y sorted by j1, j2, es)."""
    n_e = len(egrid)
    assert len(y) % n_e == 0 and np.allclose(X[:, 2].reshape(-1, n_e), egrid), "curves not on the full grid"
    return X[::n_e, :2], y.reshape(-1, n_e)

def scale_pairs(P, scalerX):
    """Scale (j1, j2) with the first two columns of scalerX."""
    return (P - scalerX.mean_[:2]) / scalerX.scale_[:2]

def predict_base_curves(P, gpr, return_std=False):
    """Base curves on gpr['egrid'] for each (j1, j2) row of P; optional 1-sigma band."""
    out = [gp.predict(scale_pairs(P, gpr['scalerX']), return_std=True) for gp in gpr['base_gps']]
    coef = np.column_stack([mu for mu, _ in out])
    curves = np.clip(gpr['mean_curve'] + coef @ gpr['components'], 0, None)
    if not return_std:
        return curves
    std = np.sqrt(np.column_stack([sd for _, sd in out]) ** 2 @ gpr['components'] ** 2)
    return curves, std

def predict_base(X, gpr):
    """Base prediction at arbitrary (j1, j2, es) rows."""
    out = np.empty(len(X))
    P = np.unique(X[:, :2], axis=0)
    for p, c in zip(P, predict_base_curves(P, gpr)):
        sel = (X[:, 0] == p[0]) & (X[:, 1] == p[1])
        out[sel] = np.interp(X[sel, 2], gpr['egrid'], c)
    return out

def predict_two_stage(X, gpr):
    """Base + GP correction at arbitrary (j1, j2, es) rows."""
    y_base = predict_base(X, gpr)
    delta = gpr['corr_gp'].predict(gpr['scalerX'].transform(X))
    return np.clip(y_base + (y_base + gpr['S']) * delta, 0, None)

if __name__ == "__main__":

    filename  = 'modelling_data.npz'
    v1, v2 = 9, 5

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 3:
        v1 = int(sys.argv[2])
        v2 = int(sys.argv[3])

    print("\n--- Loading Data ---")
    data = np.load(filename)
    Xraw, yraw = data['Xraw'], data['yraw']
    Xfit, yfit = data['Xfit'], data['yfit']
    
    print("Data shapes:")
    print(f"Raw: {Xraw.shape}, {yraw.shape} | Fit: {Xfit.shape}, {yfit.shape}")

    print(f"\n--- Processing for v1={v1}, v2={v2} ---")

    # Select indices
    selectedindex = np.where((Xraw[:, 0] == v1) & (Xraw[:, 1] == v2))
    Xraw_selected, yraw_selected = Xraw[selectedindex], yraw[selectedindex]
    
    selectedindex = np.where((Xfit[:, 0] == v1) & (Xfit[:, 1] == v2))
    Xfit_selected, yfit_selected = Xfit[selectedindex], yfit[selectedindex]
    print(f"Selected data shapes: {Xraw_selected.shape}, {yraw_selected.shape} | {Xfit_selected.shape}, {yfit_selected.shape}")     

    print(f"Fitted values < 0: {(yfit_selected < 0).sum()} (min {yfit_selected.min():.3e}) -> set to 0")
    yfit_selected = np.clip(yfit_selected, 0, None)   # the functional fit undershoots below threshold

    j1s_fit = Xfit_selected[:, 2]
    j2s_fit = Xfit_selected[:, 3]
    j1s_raw = Xraw_selected[:, 2]
    j2s_raw = Xraw_selected[:, 3]

    set_j1s_fit = set(j1s_fit)
    set_j2s_fit = set(j2s_fit)
    set_j1s_raw = set(j1s_raw)
    set_j2s_raw = set(j2s_raw)

    assert set_j1s_fit == set_j1s_raw, "Mismatch in j1 values between raw and fit data!"
    assert set_j2s_fit == set_j2s_raw, "Mismatch in j2 values between raw and fit data!"
    print("Verified that j1 and j2 values match between raw and fit datasets.")
   
    # Remove v1 and v2 features
    Xraw_selected = Xraw_selected[:, 2:]
    Xfit_selected = Xfit_selected[:, 2:]
    print(f"Selected data shapes (after removing v1,v2): {Xraw_selected.shape}, {yraw_selected.shape} | {Xfit_selected.shape}, {yfit_selected.shape}")

    # Sort both by (j1, j2, es) so row i is the same point in raw and fit
    # they shold be already aligned but we sort to be sure
    o = np.lexsort((Xraw_selected[:, 2], Xraw_selected[:, 1], Xraw_selected[:, 0]))
    Xraw_selected, yraw_selected = Xraw_selected[o], yraw_selected[o]
    o = np.lexsort((Xfit_selected[:, 2], Xfit_selected[:, 1], Xfit_selected[:, 0]))
    Xfit_selected, yfit_selected = Xfit_selected[o], yfit_selected[o]
    assert Xraw_selected.shape == Xfit_selected.shape and np.allclose(Xraw_selected, Xfit_selected), \
        "Raw and fitted data are not on the same (j1, j2, es) grid"

    # Split data
    print("\n--- Splitting Data into Train/Test ---")
    # rendom split
    pairs, pair_id = np.unique(Xraw_selected[:, :2], axis=0, return_inverse=True)
    pair_id = pair_id.ravel()
    p_trval, p_test = train_test_split(np.arange(len(pairs)), test_size=0.2, random_state=42)
    p_train, p_val = train_test_split(p_trval, test_size=0.15, random_state=42)
    m_train, m_val, m_test = (np.isin(pair_id, p) for p in (p_train, p_val, p_test))

    Xraw_selected_train, Xraw_selected_val, Xraw_selected_test = Xraw_selected[m_train], Xraw_selected[m_val], Xraw_selected[m_test]
    yraw_selected_train, yraw_selected_val, yraw_selected_test = yraw_selected[m_train], yraw_selected[m_val], yraw_selected[m_test]
    Xfit_selected_train, Xfit_selected_val, Xfit_selected_test = Xfit_selected[m_train], Xfit_selected[m_val], Xfit_selected[m_test]
    yfit_selected_train, yfit_selected_val, yfit_selected_test = yfit_selected[m_train], yfit_selected[m_val], yfit_selected[m_test]
    print(f"Curves: train {len(p_train)}, val {len(p_val)}, test {len(p_test)}")
    print("Test curves (j1, j2):", [tuple(p) for p in pairs[np.sort(p_test)]])
            
    np.savez('train_test_data.npz',
             Xraw_selected_train=Xraw_selected_train, yraw_selected_train=yraw_selected_train,
             Xraw_selected_test=Xraw_selected_test, yraw_selected_test=yraw_selected_test,
             Xfit_selected_train=Xfit_selected_train, yfit_selected_train=yfit_selected_train,
             Xfit_selected_test=Xfit_selected_test, yfit_selected_test=yfit_selected_test,
             Xfit_selected_val=Xfit_selected_val, yfit_selected_val=yfit_selected_val,
             Xraw_selected_val=Xraw_selected_val, yraw_selected_val=yraw_selected_val,
             p_train=p_train, p_val=p_val, p_test=p_test
             )

    print("\n--- Scaling Data ---")
    scalerX = StandardScaler().fit(Xraw_selected_train)
    Xraw_selected_train_scaled = scalerX.transform(Xraw_selected_train)
    Xraw_selected_test_scaled  = scalerX.transform(Xraw_selected_test)
    Xfit_selected_train_scaled = scalerX.transform(Xfit_selected_train)
    Xfit_selected_test_scaled  = scalerX.transform(Xfit_selected_test)
    Xraw_selected_val_scaled = scalerX.transform(Xraw_selected_val)
    Xfit_selected_val_scaled = scalerX.transform(Xfit_selected_val)

    # Correction target: relative residual, in ORIGINAL units
    S = 1e-2   # floor: relative correction above S, absolute below
    r_train = (yraw_selected_train - yfit_selected_train) / (yfit_selected_train + S)
    r_train = np.clip(r_train, -0.5, 0.5)

    # Save scalers
    with open('scalerX.pkl', 'wb') as fh:
        pickle.dump(scalerX, fh)

    # Build model using defined architecture
    print("\n--- STAGE 1: PCA + GP base on Fitted Data ---")
    egrid = np.unique(Xfit_selected[:, 2])                 # common energy grid
    n_e = len(egrid)

    P_train, Yfit_train = curves_matrix(Xfit_selected_train, yfit_selected_train, egrid)
    mean_curve = Yfit_train.mean(axis=0)
    _, sv, Vt = np.linalg.svd(Yfit_train - mean_curve, full_matrices=False)

    PCA_TOL = 3e-3       # max reconstruction RMSE on the training fits (raw noise is ~2e-2)
    for n_comp in range(1, len(sv) + 1):
        rec = mean_curve + (Yfit_train - mean_curve) @ Vt[:n_comp].T @ Vt[:n_comp]
        if np.sqrt(np.mean((rec - Yfit_train) ** 2)) < PCA_TOL:
            break
    components = Vt[:n_comp]
    coef_train = (Yfit_train - mean_curve) @ components.T
    print(f"PCA: {n_comp} components, reconstruction RMSE {np.sqrt(np.mean((rec - Yfit_train) ** 2)):.2e}")

    base_gps = []
    for k in range(n_comp):
        kernel = C(1.0) * Matern(length_scale=[1.0, 1.0], nu=2.5) + WhiteKernel(1e-5, (1e-10, 1e-1))
        gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=3, random_state=0)
        gp.fit(scale_pairs(P_train, scalerX), coef_train[:, k])
        base_gps.append(gp)
        print(f"  component {k}: {gp.kernel_}")

    print("\n--- STAGE 2: GP correction on raw residuals ---")
    N_PER_CURVE = 40     # raw residuals per training curve, denser at low energy
    pick = np.unique(np.round(np.linspace(0, 1, N_PER_CURVE) ** 2 * (n_e - 1)).astype(int))
    sub = (np.arange(len(P_train))[:, None] * n_e + pick[None, :]).ravel()
    print(f"GP correction on {len(sub)} points")

    kernel = C(0.1) * Matern(length_scale=[1.0, 1.0, 1.0], nu=2.5) + WhiteKernel(0.1, (1e-6, 1e1))
    corr_gp = GaussianProcessRegressor(kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0)
    corr_gp.fit(scalerX.transform(Xraw_selected_train[sub]), r_train[sub])
    k = corr_gp.kernel_
    r_std = corr_gp._y_train_std
    print(f"Correction kernel: {k}")
    print(f"  systematic amplitude {np.sqrt(k.k1.k1.constant_value) * r_std:.4f}, "
          f"noise {np.sqrt(k.k2.noise_level) * r_std:.4f} (relative units)")

    gpr = {'scalerX': scalerX, 'S': S, 'egrid': egrid, 'mean_curve': mean_curve,
           'components': components, 'base_gps': base_gps, 'corr_gp': corr_gp}
    joblib.dump(gpr, 'gpr_two_stage.joblib')