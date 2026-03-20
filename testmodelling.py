import joblib
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.callbacks import EarlyStopping


def build_model(input_dim, shapes=[512, 
                                   'BN', 
                                   256, 
                                   'BN', 
                                   256, 
                                   128, 
                                   64]):
    model = models.Sequential()
    model.add(layers.InputLayer(input_shape=(input_dim,)))
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
    yraw = np.log10(yraw)  # Log-transform the target variable for better learning
    Xfit = data['Xfit']
    yfit = data['yfit']
    yfit = np.log10(yfit)  # Log-transform the target variable for better learning
    print(Xraw.shape, yraw.shape, Xfit.shape, yfit.shape)
    
    # Split fitted data for Pre-Training
    Xfit_train, Xfit_test, yfit_train, yfit_test = train_test_split(
        Xfit, yfit, test_size=0.2, random_state=42
    )

    # Split raw data for Fine-Tuning
    Xraw_train, Xraw_test, yraw_train, yraw_test = train_test_split(
        Xraw, yraw, test_size=0.2, random_state=42
    )

    # 3. Feature Scaling (CRITICAL)
    print("Scaling features...")
    scaler = StandardScaler()
    # Fit the scaler on the fitted training data, then apply to EVERYTHING else
    Xfit_train_scaled = scaler.fit_transform(Xfit_train)
    Xfit_test_scaled = scaler.transform(Xfit_test)

    Xraw_train_scaled = scaler.transform(Xraw_train)
    Xraw_test_scaled = scaler.transform(Xraw_test)

    # 4. Build the Neural Network Architecture

    model = build_model(Xfit_train_scaled.shape[1])

    # Early stopping to prevent overfitting
    early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    # =====================================================================
    # PHASE 1: PRE-TRAINING (Learning the Physics)
    # =====================================================================
    print("\n--- PHASE 1: Pre-training on Fitted Data ---")
    # Standard learning rate for initial training
    model.compile(optimizer=optimizers.Adam(learning_rate=0.001), loss='mse')
    
    early_stop = EarlyStopping(
        monitor='val_loss',         # The metric to watch (Validation MSE)
        patience=5,                 # How many epochs to wait before stopping if no improvement
        min_delta=1e-6,             # Minimum change required to count as an "improvement"
        restore_best_weights=True,  # CRITICAL: Reverts the model to its best state!
        verbose=1                   # Prints a message when early stopping is triggered
    )

    history_fit = model.fit(
        Xfit_train_scaled, yfit_train,
        validation_data=(Xfit_test_scaled, yfit_test),
        epochs=200,
        batch_size=256,
        callbacks=[early_stop],
        verbose=1
    )

    # =====================================================================
    # PHASE 2: FINE-TUNING (Adjusting to Raw Reality)
    # =====================================================================
    print("\n--- PHASE 2: Fine-tuning on Raw Data ---")
    # Lower the learning rate drastically so we don't destroy the physics we just learned
    model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss='mse')

    history_raw = model.fit(
        Xraw_train_scaled, yraw_train,
        validation_data=(Xraw_test_scaled, yraw_test),
        epochs=200,
        batch_size=64, # Smaller batch size for smaller dataset
        callbacks=[early_stop],
        verbose=1
    )

    # =====================================================================
    # EVALUATION
    # =====================================================================
    print("\n--- Final Evaluation on Raw Test Set ---")
    loss = model.evaluate(Xraw_test_scaled, yraw_test, verbose=0)
    print(f"Final Mean Squared Error (MSE) on unseen raw data: {loss:.6e}")

    # Save the final model so you don't have to retrain it later
    model.save("cross_section_transfer_model.keras")
    print("Model saved to 'cross_section_transfer_model.keras'")

    # dump the training history for later analysis
    np.savez("training_history.npz",
             history_fit_loss=history_fit.history['loss'],
             history_fit_val_loss=history_fit.history['val_loss'],
             history_raw_loss=history_raw.history['loss'],
             history_raw_val_loss=history_raw.history['val_loss'])
    print("Training history saved to 'training_history.npz'")

    # save also test and train sets for later analysis
    np.savez("train_test_data.npz",
             Xfit_train=Xfit_train, yfit_train=yfit_train,
             Xfit_test=Xfit_test, yfit_test=yfit_test,
             Xraw_train=Xraw_train, yraw_train=yraw_train,
             Xraw_test=Xraw_test, yraw_test=yraw_test)
    print("Train/test data saved to 'train_test_data.npz'")

    # save the scaler for later use (e.g. when applying the model to new data)
    joblib.dump(scaler, "feature_scaler.joblib")
    print("Feature scaler saved to 'feature_scaler.joblib'")    
