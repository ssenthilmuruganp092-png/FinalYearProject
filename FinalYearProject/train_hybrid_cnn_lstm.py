import os
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

# 1. Dataset Loading (Smart Grid Telemetry)
def load_smart_grid_data():
    print("[+] Loading Smart Grid Telemetry Dataset...")
    
    # Load the CSV you generated
    dataset_path = 'dataset/smart_grid_telemetry.csv'
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset not found at {dataset_path}. Please run 'code generate_smart_grid_data.py' first.")
        
    df = pd.read_csv(dataset_path)
    
    # Extract the 4 physical features
    X = df[['Bus_Voltage', 'Grid_Frequency', 'Active_Power', 'Reactive_Power']].values.astype(np.float32)
    
    # Extract Labels (0 = Normal, 1 = FDI Attack)
    y = df['Label'].values.astype(np.float32)
    
    print(f"[+] Loaded {len(X)} samples with {X.shape[1]} features.")
    return X, y

# 2. Preprocessing Pipeline (Scaling only, no PCA for 4 features)
def preprocess_data(X):
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Save scaler artifact for real-time inference in app.py
    joblib.dump(scaler, "scaler.pkl")
    print("[+] Saved 'scaler.pkl' for real-time inference.")
    return X_scaled

# 3. Hybrid 1D CNN-LSTM Architecture (Adapted for Binary Classification)
def build_hybrid_model(input_shape=(4, 1)):
    model = keras.Sequential([
        keras.Input(shape=input_shape),
        
        # 1D Convolutional Layer (Spatial feature extraction)
        layers.Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'),
        layers.MaxPooling1D(pool_size=2),
        
        # LSTM Layer (Temporal sequence modeling)
        layers.LSTM(units=32, return_sequences=False),
        layers.Dropout(0.2),
        
        # Classification Dense Head
        layers.Dense(units=32, activation='relu'),
        layers.Dense(units=1, activation='sigmoid') # Sigmoid for binary (Normal vs Attack)
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy', # Binary crossentropy for 2 classes
        metrics=['accuracy']
    )
    return model

# 4. Training and TFLite Quantization Pipeline
def train_and_export_quantized():
    # Load Data
    X, y = load_smart_grid_data()
    X_scaled = preprocess_data(X)
    
    # Reshape for 1D Conv/LSTM input: (samples, 4, 1)
    X_reshaped = np.expand_dims(X_scaled, axis=-1)
    
    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)
    
    print("\n[+] Building Hybrid CNN-LSTM Model...")
    model = build_hybrid_model(input_shape=(4, 1))
    model.summary()
    
    # Train Model
    print("\n[+] Training Hybrid CNN-LSTM Model...")
    model.fit(
        X_train, y_train,
        epochs=10, # Increased epochs slightly for better convergence
        batch_size=64,
        validation_split=0.1,
        verbose=1
    )
    
    # Evaluate on Test Set
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[+] Test Set Evaluation -> Loss: {test_loss:.4f} | Accuracy: {test_acc*100:.2f}%")
    
    # Save Native Keras Model
    model_save_path = "hybrid_cnn_lstm.keras"
    model.save(model_save_path)
    print(f"[+] Saved Model: {model_save_path}")
    
    # Post-Training FP16 / Dynamic Range Quantization (TFLite Converter)
    print("\n[+] Executing Post-Training Quantization (TFLite Converter)...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    # Enable TF select ops for LSTM support
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter._experimental_lower_tensor_list_ops = False
    
    tflite_quant_model = converter.convert()
    tflite_path = "hybrid_cnn_lstm_quantized.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)
    
    print("\n" + "=" * 60)
    print("[✓] Model Training & Quantization Complete!")
    print(f"    - Original Model Size: {os.path.getsize(model_save_path) / 1024:.2f} KB")
    print(f"    - Quantized TFLite Size: {os.path.getsize(tflite_path) / 1024:.2f} KB")
    print(f"    - Compression Ratio: {(os.path.getsize(model_save_path) / os.path.getsize(tflite_path)):.2f}x")
    print("=" * 60)

if __name__ == "__main__":
    train_and_export_quantized()