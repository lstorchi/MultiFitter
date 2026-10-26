import sys
import joblib
import pickle
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

# Standardized TensorFlow Keras imports
#from tensorflow.keras import layers, models, optimizers
#from tensorflow.keras.callbacks import Early

from keras import layers, models, optimizers
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def make_early_stop():
    return EarlyStopping(monitor='val_loss', patience=25, min_delta=1e-6,
         restore_best_weights=True, verbose=1)

def build_model(input_dim, shapes=[64, 64, 'BN', 32, 32, 16, 8], 
                out_activation='softplus'):
    model = models.Sequential()
    model.add(layers.Input(shape=(input_dim,)))
    
    for shape in shapes:
        if shape == 'BN':
            model.add(layers.BatchNormalization())
        else:
            model.add(layers.Dense(shape, activation='relu'))
            
    model.add(layers.Dense(1, activation=out_activation))  # 'softplus' keeps predictions >= 0
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
    
    #Xfit_selected_train, Xfit_selected_val, \
    #Xraw_selected_train, Xraw_selected_val, \
    #yfit_selected_train, yfit_selected_val, \
    #yraw_selected_train, yraw_selected_val = train_test_split(
    #        Xfit_selected_train, Xraw_selected_train, yfit_selected_train, yraw_selected_train,
    #        test_size=0.15, random_state=42)
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

    # Save scalers
    with open('scalerX.pkl', 'wb') as fh:
        pickle.dump(scalerX, fh)
    with open('scalery.pkl', 'wb') as fh:
        pickle.dump(scalery, fh)

    # Build model using defined architecture
    print("\n--- PHASE 1: Pre-training on Fitted Data ---")
    model_architecture = [64, 64]
    model = build_model(Xfit_selected_train_scaled.shape[1], shapes=model_architecture, 
                            out_activation='softplus')
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    history_fit = model.fit(
            Xfit_selected_train_scaled, yfit_selected_train_scaled,
            validation_data=(Xfit_selected_val_scaled, \
                             yfit_selected_val_scaled),
            epochs=500,
            batch_size=256,
            callbacks=[make_early_stop()],
            verbose=1
        )

    joblib.dump(model, 'pretrained_model.joblib')
    joblib.dump(history_fit.history, 'pretraining_history.joblib')

    print("\n--- PHASE 2: Fine-tuning on Raw Data (Partial Freezing) ---")
    # 1. Freeze the early layers
    n_trainable = 2   # layers left trainable, counted from the output (1 = output layer only)
    for layer in model.layers[:-n_trainable]:
        layer.trainable = False
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')

    # (Optional) Print a quick summary to verify which layers are locked
    for i, layer in enumerate(model.layers):
        status = "Trainable" if layer.trainable else "FROZEN"
        print(f"Layer {i} ({layer.name}): {status}")
    
    history_raw = model.fit(
            Xraw_selected_train_scaled, yraw_selected_train_scaled,
            validation_data=(Xraw_selected_val_scaled, \
                             yraw_selected_val_scaled),
            epochs=500,
            batch_size=64, 
            callbacks=[make_early_stop()],
            verbose=1
        )
        
    joblib.dump(model, 'fine_tuned_model.joblib')
    joblib.dump(history_raw.history, 'fine_tuning_history.joblib')
