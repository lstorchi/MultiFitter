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

def build_model(input_dim, shapes=[64, 64, 'BN', 32, 32, 16, 8]):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
    
    for shape in shapes:
        if shape == 'BN':
            model.add(layers.BatchNormalization())
        else:
            model.add(layers.Dense(shape, activation='relu'))
            
    model.add(layers.Dense(1))  # Single linear output for regression
    
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
    # split by j
    #j1s_train, j1s_test = train_test_split(list(set_j1s_raw), test_size=0.2, random_state=42)
    #j2s_train, j2s_test = train_test_split(list(set_j2s_raw), test_size=0.2, random_state=42)
    #raw_train_index = np.isin(Xraw_selected[:, 0], j1s_train) & np.isin(Xraw_selected[:, 1], j2s_train)
    #raw_test_index = np.isin(Xraw_selected[:, 0], j1s_test) & np.isin(Xraw_selected[:, 1], j2s_test)
    #Xraw_selected_train, Xraw_selected_test = Xraw_selected[raw_train_index], Xraw_selected[raw_test_index] 
    #yraw_selected_train, yraw_selected_test = yraw_selected[raw_train_index], yraw_selected[raw_test_index] 
    #fit_train_index = np.isin(Xfit_selected[:, 0], j1s_train) & np.isin(Xfit_selected[:, 1], j2s_train)
    #fit_test_index = np.isin(Xfit_selected[:, 0], j1s_test) & np.isin(Xfit_selected[:, 1], j2s_test)    
    #Xfit_selected_train, Xfit_selected_test = Xfit_selected[fit_train_index], Xfit_selected[fit_test_index]
    #yfit_selected_train, yfit_selected_test = yfit_selected[fit_train_index], yfit_selected[fit_test_index]
    # rendom split
    Xfit_selected_train, Xfit_selected_test, yfit_selected_train, yfit_selected_test = train_test_split(
            Xfit_selected, yfit_selected, test_size=0.2, random_state=42
        )
    Xraw_selected_train, Xraw_selected_test, yraw_selected_train, yraw_selected_test = train_test_split(
            Xraw_selected, yraw_selected, test_size=0.2, random_state=42
        )

    np.savez('train_test_data.npz',
             Xraw_selected_train=Xraw_selected_train, yraw_selected_train=yraw_selected_train,
             Xraw_selected_test=Xraw_selected_test, yraw_selected_test=yraw_selected_test,
             Xfit_selected_train=Xfit_selected_train, yfit_selected_train=yfit_selected_train,
             Xfit_selected_test=Xfit_selected_test, yfit_selected_test=yfit_selected_test,
             )

    print("\n--- Scaling Data ---")
    scalerX = StandardScaler().fit(Xraw_selected_train)
    Xraw_selected_train_scaled = scalerX.transform(Xraw_selected_train)
    Xraw_selected_test_scaled  = scalerX.transform(Xraw_selected_test)
    Xfit_selected_train_scaled = scalerX.transform(Xfit_selected_train)
    Xfit_selected_test_scaled  = scalerX.transform(Xfit_selected_test)

    scalery = StandardScaler(with_mean=False).fit(yfit_selected_train.reshape(-1, 1))
    yfit_selected_train_scaled = scalery.transform(yfit_selected_train.reshape(-1, 1)).ravel()
    yfit_selected_test_scaled  = scalery.transform(yfit_selected_test.reshape(-1, 1)).ravel()
    yraw_selected_train_scaled = scalery.transform(yraw_selected_train.reshape(-1, 1)).flatten()
    yraw_selected_test_scaled = scalery.transform(yraw_selected_test.reshape(-1, 1)).flatten()

    # Save scalers
    pickle.dump(scalerX, open('scalerX.pkl', 'wb'))
    pickle.dump(scalery, open('scalery.pkl', 'wb'))

    # Build model using defined architecture
    print("\n--- PHASE 1: Pre-training on Fitted Data ---")
    model_architecture = [64, 64]
    model = build_model(Xfit_selected_train_scaled.shape[1], shapes=model_architecture)
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')

    early_stop = EarlyStopping(
            monitor='val_loss',         
            patience=5,                 
            min_delta=1e-6,             
            restore_best_weights=True,  
            verbose=1                   
        )

    history_fit = model.fit(
            Xfit_selected_train_scaled, yfit_selected_train_scaled,
            validation_data=(Xfit_selected_test_scaled, yfit_selected_test_scaled),
            epochs=200,
            batch_size=256,
            callbacks=[early_stop],
            verbose=1
        )

    joblib.dump(model, 'pretrained_model.joblib')
    joblib.dump(history_fit.history, 'pretraining_history.joblib')

    print("\n--- PHASE 2: Fine-tuning on Raw Data (Partial Freezing) ---")
    # 1. Freeze the early layers
    # We will leave only the last layer trainable (the final hidden Dense layer and the Output layer)
    for layer in model.layers[:-2]:  # Freeze all layers except the last one
        layer.trainable = False
    
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')

    # (Optional) Print a quick summary to verify which layers are locked
    for i, layer in enumerate(model.layers):
        status = "Trainable" if layer.trainable else "FROZEN"
        print(f"Layer {i} ({layer.name}): {status}")
    
    history_raw = model.fit(
            Xraw_selected_train_scaled, yraw_selected_train_scaled,
            validation_data=(Xraw_selected_test_scaled, yraw_selected_test_scaled),
            epochs=200,
            batch_size=64, 
            callbacks=[early_stop],
            verbose=1
        )
        
    joblib.dump(model, 'fine_tuned_model.joblib')
    joblib.dump(history_raw.history, 'fine_tuning_history.joblib')
