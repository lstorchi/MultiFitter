import sys
import joblib
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Standardized TensorFlow Keras imports
#from tensorflow.keras import layers, models, optimizers
#from tensorflow.keras.callbacks import Early

from keras import layers, models, optimizers, regularizers, losses
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_early_stop():
    return EarlyStopping(monitor='val_loss', patience=25, min_delta=1e-6,
         restore_best_weights=True, verbose=1)

def build_model(input_dim, shapes=[64, 64, 'BN', 32, 32, 16, 8],
                out_activation='softplus', l2=0.0, zero_output=False):
    reg = regularizers.l2(l2) if l2 > 0 else None
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))

    for shape in shapes:
        if shape == 'BN':
            model.add(layers.BatchNormalization())
        else:
            model.add(layers.Dense(shape, activation='relu', kernel_regularizer=reg))

    init = 'zeros' if zero_output else 'glorot_uniform'
    model.add(layers.Dense(1, activation=out_activation,
                           kernel_initializer=init, bias_initializer='zeros'))
    return model

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
    Xfit_selected_train, Xfit_selected_test, \
    Xraw_selected_train, Xraw_selected_test, \
    yfit_selected_train, yfit_selected_test, \
    yraw_selected_train, yraw_selected_test = train_test_split(
            Xfit_selected, Xraw_selected, yfit_selected, yraw_selected, \
                test_size=0.2, random_state=42
        )
    
    Xfit_selected_train, Xfit_selected_val, \
    Xraw_selected_train, Xraw_selected_val, \
    yfit_selected_train, yfit_selected_val, \
    yraw_selected_train, yraw_selected_val = train_test_split(
            Xfit_selected_train, Xraw_selected_train, yfit_selected_train, yraw_selected_train,
            test_size=0.15, random_state=42)
            
    np.savez('train_test_data.npz',
             Xraw_selected_train=Xraw_selected_train, yraw_selected_train=yraw_selected_train,
             Xraw_selected_test=Xraw_selected_test, yraw_selected_test=yraw_selected_test,
             Xfit_selected_train=Xfit_selected_train, yfit_selected_train=yfit_selected_train,
             Xfit_selected_test=Xfit_selected_test, yfit_selected_test=yfit_selected_test,
             Xfit_selected_val=Xfit_selected_val, yfit_selected_val=yfit_selected_val,
             Xraw_selected_val=Xraw_selected_val, yraw_selected_val=yraw_selected_val
             )

    print("\n--- Scaling Data ---")
    scalerX = StandardScaler().fit(Xraw_selected_train)
    Xraw_selected_train_scaled = scalerX.transform(Xraw_selected_train)
    Xraw_selected_test_scaled  = scalerX.transform(Xraw_selected_test)
    Xfit_selected_train_scaled = scalerX.transform(Xfit_selected_train)
    Xfit_selected_test_scaled  = scalerX.transform(Xfit_selected_test)
    Xraw_selected_val_scaled = scalerX.transform(Xraw_selected_val)
    Xfit_selected_val_scaled = scalerX.transform(Xfit_selected_val)

    scalery = StandardScaler(with_mean=False).fit(yfit_selected_train.reshape(-1, 1))
    yfit_selected_train_scaled = scalery.transform(yfit_selected_train.reshape(-1, 1)).ravel()
    yfit_selected_test_scaled  = scalery.transform(yfit_selected_test.reshape(-1, 1)).ravel()
    yraw_selected_train_scaled = scalery.transform(yraw_selected_train.reshape(-1, 1)).flatten()
    yraw_selected_test_scaled = scalery.transform(yraw_selected_test.reshape(-1, 1)).flatten()
    yfit_selected_val_scaled = scalery.transform(yfit_selected_val.reshape(-1, 1)).ravel()
    yraw_selected_val_scaled = scalery.transform(yraw_selected_val.reshape(-1, 1)).ravel()
    
    # Correction target: relative residual, in ORIGINAL units
    S = 1e-2   # floor: relative correction above S, absolute below
    r_train = (yraw_selected_train - yfit_selected_train) / (yfit_selected_train + S)
    r_val   = (yraw_selected_val   - yfit_selected_val)   / (yfit_selected_val   + S)
    r_train = np.clip(r_train, -0.5, 0.5)
    r_val   = np.clip(r_val,   -0.5, 0.5)

    scaler_r = StandardScaler(with_mean=False).fit(r_train.reshape(-1, 1))   # keeps 0 at 0
    r_train_scaled = scaler_r.transform(r_train.reshape(-1, 1)).ravel()
    r_val_scaled   = scaler_r.transform(r_val.reshape(-1, 1)).ravel()
    
    # Save scalers
    with open('scalerX.pkl', 'wb') as fh:
        pickle.dump(scalerX, fh)
    with open('scalery.pkl', 'wb') as fh:
        pickle.dump(scalery, fh)
    with open('scaler_r.pkl', 'wb') as fh:
        pickle.dump(scaler_r, fh)

    # Build model using defined architecture
    print("\n--- STAGE 1: Base network on Fitted Data ---")
    model_architecture = [64, 64]
    base_model = build_model(Xfit_selected_train_scaled.shape[1], shapes=model_architecture,
                             out_activation='softplus')
    base_model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    history_fit = base_model.fit(
        Xfit_selected_train_scaled, yfit_selected_train_scaled,
        validation_data=(Xfit_selected_val_scaled, yfit_selected_val_scaled),
        epochs=500,
        batch_size=256,
        callbacks=[make_early_stop()],
        verbose=1
    )

    joblib.dump(base_model, 'base_model.joblib')
    joblib.dump(history_fit.history, 'base_history.joblib')

    print("\n--- STAGE 2: Correction network on raw residuals ---")
    corr_architecture = [32, 32]
    corr_model = build_model(Xraw_selected_train_scaled.shape[1], shapes=corr_architecture,
                             out_activation=None, l2=1e-5, zero_output=True)
    corr_model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss=losses.Huber())

    history_corr = corr_model.fit(
            Xraw_selected_train_scaled, r_train_scaled,
            validation_data=(Xraw_selected_val_scaled, r_val_scaled),
            epochs=500,
            batch_size=256,
            callbacks=[make_early_stop()],
            verbose=1
        )

    joblib.dump(corr_model, 'correction_model.joblib')
    joblib.dump(history_corr.history, 'correction_history.joblib')