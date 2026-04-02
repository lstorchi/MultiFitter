import joblib
import pickle
import numpy as np
import tensorflow as tf

import matplotlib.pyplot as plt

#from tensorflow.keras import layers, models, optimizers
from keras import layers, models, optimizers
#from tensorflow.keras.callbacks import EarlyStopping
from keras.callbacks import EarlyStopping

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def build_model(input_dim, shapes=[512, 
                                   'BN', 
                                   256, 
                                   'BN', 
                                   256, 
                                   128, 
                                   64]):
    model = models.Sequential()
    model.add(layers.InputLayer(shape=(input_dim,)))
    for shape in shapes:
        if shape == 'BN':
            model.add(layers.BatchNormalization())
        else:
            model.add(layers.Dense(shape, activation='relu'))
    model.add(layers.Dense(1))  # Single linear output for regression (cross-section)
    return model

if __name__ == "__main__":

    data = np.load('modelling_data.npz')
    Xraw = data['Xraw']
    yraw = data['yraw']
    #yraw_safe = np.clip(data['yraw'], a_min=1e-30, a_max=None)
    #yraw = np.log10(yraw_safe)
    Xfit = data['Xfit']
    yfit = data['yfit']
    #yfit_safe = np.clip(data['yfit'], a_min=1e-30, a_max=None)
    #yfit = np.log10(yfit_safe)
    print("Data shapes:")
    print(Xraw.shape, yraw.shape, Xfit.shape, yfit.shape)

    v1 = 10
    v2 = 0
    print(f"\n--- Processing for v1={v1}, v2={v2} ---")

    selectedindex = np.where((Xraw[:, 0] == v1) & (Xraw[:, 1] == v2))
    Xraw_selected = Xraw[selectedindex]
    yraw_selected = yraw[selectedindex]
    print("Selected data shapes:")
    print(Xraw_selected.shape, yraw_selected.shape)
    Xfit_selected = Xfit[selectedindex]
    yfit_selected = yfit[selectedindex]
    print("Selected fitted data shapes:")
    print(Xfit_selected.shape, yfit_selected.shape)     
    Xraw_selected = Xraw_selected[:, 2:]
    Xfit_selected = Xfit_selected[:, 2:]
    print("Selected data shapes after removing v1 and v2:")
    print(Xraw_selected.shape, Xfit_selected.shape)

    # Split fitted data for Pre-Training
    Xfit_selected_train, Xfit_selected_test, yfit_selected_train, yfit_selected_test = train_test_split(
            Xfit_selected, yfit_selected, test_size=0.2, random_state=42
        )

    # Split raw data for Fine-Tuning
    Xraw_selected_train, Xraw_selected_test, yraw_selected_train, yraw_selected_test = train_test_split(
            Xraw_selected, yraw_selected, test_size=0.2, random_state=42
        )
    print("Training shapes:")
    print(Xfit_selected_train.shape, yfit_selected_train.shape)
    print(Xraw_selected_train.shape, yraw_selected_train.shape)
    print("Test shapes:")
    print(Xfit_selected_test.shape, yfit_selected_test.shape)
    print(Xraw_selected_test.shape, yraw_selected_test.shape)

    print("\n--- Scaling Data ---")
    scalerXraw = StandardScaler()
    Xraw_selected_train_scaled = scalerXraw.fit_transform(Xraw_selected_train)
    Xraw_selected_test_scaled = scalerXraw.transform(Xraw_selected_test)
    scaleryraw = StandardScaler()
    yraw_selected_train_scaled = scaleryraw.fit_transform(yraw_selected_train.reshape(-1, 1)).flatten()
    yraw_selected_test_scaled = scaleryraw.transform(yraw_selected_test.reshape(-1, 1)).flatten()
    scalerXfit = StandardScaler()
    Xfit_selected_train_scaled = scalerXfit.fit_transform(Xfit_selected_train)
    Xfit_selected_test_scaled = scalerXfit.transform(Xfit_selected_test)
    scaleryfit = StandardScaler()
    yfit_selected_train_scaled = scaleryfit.fit_transform(yfit_selected_train.reshape(-1, 1)).flatten()
    yfit_selected_test_scaled = scaleryfit.transform(yfit_selected_test.reshape(-1, 1)).flatten()       
    print("Scaled training shapes:")
    print(Xfit_selected_train_scaled.shape, yfit_selected_train_scaled.shape)
    print(Xraw_selected_train_scaled.shape, yraw_selected_train_scaled.shape)
    print("Scaled test shapes:")
    print(Xfit_selected_test_scaled.shape, yfit_selected_test_scaled.shape)     
    print(Xraw_selected_test_scaled.shape, yraw_selected_test_scaled.shape)
    print("Feature means after scaling (should be ~0):")
    print("Fitted data:", np.mean(Xfit_selected_train_scaled, axis=0))
    print("Raw data:", np.mean(Xraw_selected_train_scaled, axis=0))
    print("Feature stds after scaling (should be ~1):")
    print("Fitted data:", np.std(Xfit_selected_train_scaled, axis=0))
    print("Raw data:", np.std(Xraw_selected_train_scaled, axis=0))
    pickle.dump(scalerXraw, open('scalerXraw.pkl', 'wb'))
    pickle.dump(scaleryraw, open('scaleryraw.pkl', 'wb'))
    pickle.dump(scalerXfit, open('scalerXfit.pkl', 'wb'))
    pickle.dump(scaleryfit, open('scaleryfit.pkl', 'wb'))

    shapes=[128, 'BN', 34, 'BN', 32, 12]
    shapes=[512, 'BN', 256, 'BN', 256, 128, 64]
    model = build_model(Xfit_selected_train_scaled.shape[1])

    print("\n--- PHASE 1: Pre-training on Fitted Data ---")
    # Standard learning rate for initial training
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001),
                   loss='mse')

    early_stop = EarlyStopping(
            monitor='val_loss',         # The metric to watch (Validation MSE)
            patience=5,                 # How many epochs to wait before stopping if no improvement
            min_delta=1e-6,             # Minimum change required to count as an "improvement"
            restore_best_weights=True,  # CRITICAL: Reverts the model to its best state!
            verbose=1                   # Prints a message when early stopping is triggered
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

    print("\n--- PHASE 2: Fine-tuning on Raw Data ---")
    # Lower the learning rate drastically so we don't destroy the physics we just learned
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')

    history_raw = model.fit(
            Xraw_selected_train_scaled, yraw_selected_train,
            validation_data=(Xraw_selected_test_scaled, yraw_selected_test),
            epochs=200,
            batch_size=64, # Smaller batch size for smaller dataset
            callbacks=[early_stop],
            verbose=1
        )
    joblib.dump(model, 'fine_tuned_model.joblib')
    joblib.dump(history_raw.history, 'fine_tuning_history.joblib')

    # predict for training and test and plot
    yfit_train_pred_scaled = model.predict(Xfit_selected_train_scaled).flatten()
    yfit_test_pred_scaled = model.predict(Xfit_selected_test_scaled).flatten()
    yfit_train_pred = scaleryfit.inverse_transform(yfit_train_pred_scaled.reshape(-1, 1)).flatten()
    yfit_test_pred = scaleryfit.inverse_transform(yfit_test_pred_scaled.reshape(-1, 1)).flatten()
    rmse_fit_train = np.sqrt(np.mean((yfit_train_pred - yfit_selected_train)**2))
    mape_fit_train = np.mean(np.abs((yfit_selected_train - yfit_train_pred) / yfit_selected_train)) * 100
    rmse_fit_test = np.sqrt(np.mean((yfit_test_pred - yfit_selected_test)**2))
    mape_fit_test = np.mean(np.abs((yfit_selected_test - yfit_test_pred) / yfit_selected_test)) * 100
    print(f"Fitted Data - Train RMSE: {rmse_fit_train:.4f}, Test RMSE: {rmse_fit_test:.4f}")
    print(f"Fitted Data - Train MAPE: {mape_fit_train:.2f}%, Test MAPE: {mape_fit_test:.2f}%")
    yraw_train_pred_scaled = model.predict(Xraw_selected_train_scaled).flatten()
    yraw_test_pred_scaled = model.predict(Xraw_selected_test_scaled).flatten()
    yraw_train_pred = scaleryraw.inverse_transform(yraw_train_pred_scaled.reshape(-1, 1)).flatten()
    yraw_test_pred = scaleryraw.inverse_transform(yraw_test_pred_scaled.reshape(-1, 1)).flatten()   
    rmse_raw_train = np.sqrt(np.mean((yraw_train_pred - yraw_selected_train)**2))
    mape_raw_train = np.mean(np.abs((yraw_selected_train - yraw_train_pred) / yraw_selected_train)) * 100
    rmse_raw_test = np.sqrt(np.mean((yraw_test_pred - yraw_selected_test)**2))
    mape_raw_test = np.mean(np.abs((yraw_selected_test - yraw_test_pred) / yraw_selected_test)) * 100
    print(f"Raw Data - Train RMSE: {rmse_raw_train:.4f}, Test RMSE: {rmse_raw_test:.4f}")
    print(f"Raw Data - Train MAPE: {mape_raw_train:.2f}%, Test MAPE: {mape_raw_test:.2f}%")
