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

def build_model(input_dim, shapes=[512, 'BN', 256, 'BN', 256, 128, 64]):
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

    print("\n--- Loading Data ---")
    data = np.load('modelling_data.npz')
    Xraw, yraw = data['Xraw'], data['yraw']
    Xfit, yfit = data['Xfit'], data['yfit']
    
    print("Data shapes:")
    print(f"Raw: {Xraw.shape}, {yraw.shape} | Fit: {Xfit.shape}, {yfit.shape}")

    v1, v2 = 10, 0
    print(f"\n--- Processing for v1={v1}, v2={v2} ---")

    # Select indices
    selectedindex = np.where((Xraw[:, 0] == v1) & (Xraw[:, 1] == v2))
    Xraw_selected, yraw_selected = Xraw[selectedindex], yraw[selectedindex]
    
    selectedindex = np.where((Xfit[:, 0] == v1) & (Xfit[:, 1] == v2))
    Xfit_selected, yfit_selected = Xfit[selectedindex], yfit[selectedindex]

    print(f"Selected data shapes: {Xraw_selected.shape}, {yraw_selected.shape} | {Xfit_selected.shape}, {yfit_selected.shape}")     


    # select all value lower than 1e-9 and remove them both from y and X
    print("\n--- Filtering out values < 1e-2 ---")
    mask_raw = yraw_selected >= 1e-2
    mask_fit = yfit_selected >= 1e-2
    Xraw_selected = Xraw_selected[mask_raw]
    yraw_selected = yraw_selected[mask_raw]
    Xfit_selected = Xfit_selected[mask_fit]
    yfit_selected = yfit_selected[mask_fit]
    print(f"Data shapes after filtering out values < 1e-2: {Xraw_selected.shape}, {yraw_selected.shape} | {Xfit_selected.shape}, {yfit_selected.shape}")


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
   
    plt.figure(figsize=(24, 10))
    minvraw = float('inf')
    minvfit = float('inf')
    maxvraw = float('-inf')
    maxvfit = float('-inf')
    for j1 in set_j1s_raw:
        for j2 in set_j2s_raw:
            selectedindex = np.where((Xraw_selected[:, 2] == j1) & (Xraw_selected[:, 3] == j2))
            Xraw_subset = Xraw_selected[selectedindex]
            yraw_subset = yraw_selected[selectedindex]
            if np.min(yraw_subset) < minvraw:
                minvraw = np.min(yraw_subset)
            if np.max(yraw_subset) > maxvraw:
                maxvraw = np.max(yraw_subset)
            esraw = Xraw_subset[:, 4]
            selectedindex = np.where((Xfit_selected[:, 2] == j1) & (Xfit_selected[:, 3] == j2))
            Xfit_subset = Xfit_selected[selectedindex]
            yfit_subset = yfit_selected[selectedindex]
            if np.min(yfit_subset) < minvfit:
                minvfit = np.min(yfit_subset)
            if np.max(yfit_subset) > maxvfit:
                maxvfit = np.max(yfit_subset)
            esfit = Xfit_subset[:, 4]
            plt.scatter(esraw, yraw_subset, alpha=0.5, color='orange', label='Raw Data')
            plt.scatter(esfit, yfit_subset, linewidths=1, color='blue', label='Fitted Data')   
    plt.xlabel('es')
    plt.ylabel('y')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('es_vs_y.png')
    plt.close()
    print(f"Observed y ranges - Raw: [{minvraw:.2e}, {maxvraw:.2e}], Fit: [{minvfit:.2e}, {maxvfit:.2e}]")

    # Remove v1 and v2 features
    Xraw_selected = Xraw_selected[:, 2:]
    Xfit_selected = Xfit_selected[:, 2:]
    print(f"Selected data shapes (after removing v1,v2): {Xraw_selected.shape}, {yraw_selected.shape} | {Xfit_selected.shape}, {yfit_selected.shape}")


    print("\n--- Data Preprocessing ---")
    yraw_selected = np.log10(yraw_selected)
    yfit_selected = np.log10(yfit_selected)
    print("Applied log10 transformation to targets.")

    # plot es vs log10(y) to verify the transformation
    plt.figure(figsize=(24, 10))
    minvraw_log = float('inf')
    maxvraw_log = float('-inf')
    minvfit_log = float('inf')
    maxvfit_log = float('-inf')
    for j1 in set_j1s_raw:
        for j2 in set_j2s_raw:
            selectedindex = np.where((Xraw_selected[:, 0] == j1) & (Xraw_selected[:, 1] == j2))
            Xraw_subset = Xraw_selected[selectedindex]
            yraw_subset = yraw_selected[selectedindex]
            esraw = Xraw_subset[:, 2]
            selectedindex = np.where((Xfit_selected[:, 0] == j1) & (Xfit_selected[:, 1] == j2))
            Xfit_subset = Xfit_selected[selectedindex]
            yfit_subset = yfit_selected[selectedindex]
            esfit = Xfit_subset[:, 2]
            if np.min(yraw_subset) < minvraw_log:
                minvraw_log = np.min(yraw_subset)
            if np.max(yraw_subset) > maxvraw_log:
                maxvraw_log = np.max(yraw_subset)
            if np.min(yfit_subset) < minvfit_log:
                minvfit_log = np.min(yfit_subset)
            if np.max(yfit_subset) > maxvfit_log:
                maxvfit_log = np.max(yfit_subset)
            plt.scatter(esraw, yraw_subset, alpha=0.5, color='orange', label='Raw Data (log10)')
            plt.scatter(esfit, yfit_subset, linewidths=1, color='blue', label='Fitted Data (log10)')
    plt.xlabel('es')
    plt.ylabel('log10(y)')
    #plt.legend()
    plt.tight_layout()
    plt.savefig('es_vs_log10y.png')
    plt.close()
    print(f"Observed log10(y) ranges - Raw: [{minvraw_log:.2e}, {maxvraw_log:.2e}], Fit: [{minvfit_log:.2e}, {maxvfit_log:.2e}]")

    # Split data
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

    # Save scalers
    pickle.dump(scalerXraw, open('scalerXraw.pkl', 'wb'))
    pickle.dump(scaleryraw, open('scaleryraw.pkl', 'wb'))
    pickle.dump(scalerXfit, open('scalerXfit.pkl', 'wb'))
    pickle.dump(scaleryfit, open('scaleryfit.pkl', 'wb'))

    # Build model using defined architecture
    print("\n--- PHASE 1: Pre-training on Fitted Data ---")
    model_architecture = [512, 'BN', 256, 'BN', 256, 128, 64]
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
    # We will leave only the last two layers trainable (the final hidden Dense layer and the Output layer)
    #for layer in model.layers[:-2]:
    #    layer.trainable = False
    
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

    print("\n--- Evaluating Model ---")
    epsilon = 1e-10  # Prevent divide-by-zero in MAPE

    # Fitted Data Eval
    yfit_train_pred_scaled = model.predict(Xfit_selected_train_scaled, verbose=0).flatten()
    yfit_test_pred_scaled = model.predict(Xfit_selected_test_scaled, verbose=0).flatten()
    
    yfit_train_pred = scaleryfit.inverse_transform(yfit_train_pred_scaled.reshape(-1, 1)).flatten()
    yfit_test_pred = scaleryfit.inverse_transform(yfit_test_pred_scaled.reshape(-1, 1)).flatten()
    
    rmse_fit_train = np.sqrt(np.mean((yfit_train_pred - yfit_selected_train)**2))
    mape_fit_train = np.mean(np.abs((yfit_selected_train - yfit_train_pred) / (yfit_selected_train + epsilon))) * 100
    rmse_fit_test = np.sqrt(np.mean((yfit_test_pred - yfit_selected_test)**2))
    mape_fit_test = np.mean(np.abs((yfit_selected_test - yfit_test_pred) / (yfit_selected_test + epsilon))) * 100
    mae_fit_train = np.mean(np.abs(yfit_selected_train - yfit_train_pred))
    mae_fit_test = np.mean(np.abs(yfit_selected_test - yfit_test_pred))
    r2_fit_train = 1 - np.sum((yfit_selected_train - yfit_train_pred)**2) / np.sum((yfit_selected_train - np.mean(yfit_selected_train))**2)
    r2_fit_test = 1 - np.sum((yfit_selected_test - yfit_test_pred)**2) / np.sum((yfit_selected_test - np.mean(yfit_selected_test))**2)
    print(f"Fitted Data - Train RMSE: {rmse_fit_train:.4f}, Test RMSE: {rmse_fit_test:.4f}")
    print(f"Fitted Data - Train MAPE: {mape_fit_train:.2f}%, Test MAPE: {mape_fit_test:.2f}%")
    print(f"Fitted Data - Train MAE: {mae_fit_train:.4f}, Test MAE: {mae_fit_test:.4f}")
    print(f"Fitted Data - Train R2: {r2_fit_train:.4f}, Test R2: {r2_fit_test:.4f}")

    # Raw Data Eval
    yraw_train_pred_scaled = model.predict(Xraw_selected_train_scaled, verbose=0).flatten()
    yraw_test_pred_scaled = model.predict(Xraw_selected_test_scaled, verbose=0).flatten()
    
    yraw_train_pred = scaleryraw.inverse_transform(yraw_train_pred_scaled.reshape(-1, 1)).flatten()
    yraw_test_pred = scaleryraw.inverse_transform(yraw_test_pred_scaled.reshape(-1, 1)).flatten()   
    
    rmse_raw_train = np.sqrt(np.mean((yraw_train_pred - yraw_selected_train)**2))
    mape_raw_train = np.mean(np.abs((yraw_selected_train - yraw_train_pred) / (yraw_selected_train + epsilon))) * 100
    mae_raw_train = np.mean(np.abs(yraw_selected_train - yraw_train_pred))
    r2_raw_train = 1 - np.sum((yraw_selected_train - yraw_train_pred)**2) / np.sum((yraw_selected_train - np.mean(yraw_selected_train))**2)
    rmse_raw_test = np.sqrt(np.mean((yraw_test_pred - yraw_selected_test)**2))
    mape_raw_test = np.mean(np.abs((yraw_selected_test - yraw_test_pred) / (yraw_selected_test + epsilon))) * 100
    mae_raw_test = np.mean(np.abs(yraw_selected_test - yraw_test_pred))
    r2_raw_test = 1 - np.sum((yraw_selected_test - yraw_test_pred)**2) / np.sum((yraw_selected_test - np.mean(yraw_selected_test))**2)
    
    print(f"Raw Data - Train RMSE: {rmse_raw_train:.4f}, Test RMSE: {rmse_raw_test:.4f}")
    print(f"Raw Data - Train MAPE: {mape_raw_train:.2f}%, Test MAPE: {mape_raw_test:.2f}%")
    print(f"Raw Data - Train MAE: {mae_raw_train:.4f}, Test MAE: {mae_raw_test:.4f}")
    print(f"Raw Data - Train R2: {r2_raw_train:.4f}, Test R2: {r2_raw_test:.4f}")