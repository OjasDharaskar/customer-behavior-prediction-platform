import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "dl_training.log"), mode="w")
    ]
)
logger = logging.getLogger(__name__)

def load_data(data_path: str):
    logger.info(f"Loading processed dataset from {data_path}...")
    df = pd.read_csv(data_path)
    
    # Define targets
    targets = {
        "churn": df["target_churn"],
        "clv": df["target_clv"],
        "next_category": df["target_next_category"]
    }
    
    # Feature columns (exclude customer_id and targets)
    feature_cols = [
        col for col in df.columns 
        if col not in ["customer_id", "target_churn", "target_clv", "target_next_category"]
    ]
    
    X = df[feature_cols].copy()
    logger.info(f"Dataset loaded. Shape: {df.shape}. Number of features: {len(feature_cols)}.")
    return X, targets, df

def split_data(X, y_churn, y_clv, y_next):
    logger.info("Splitting dataset into train (70%), validation (15%), and test (15%)...")
    
    # Split 85% train_val and 15% test
    X_train_val, X_test, y_churn_train_val, y_churn_test, y_clv_train_val, y_clv_test, y_next_train_val, y_next_test = train_test_split(
        X, y_churn, y_clv, y_next, test_size=0.15, random_state=42, stratify=y_churn
    )
    
    # Split train_val (85%) into train (70% total) and validation (15% total)
    # Ratio: 15 / 85 = 0.17647
    X_train, X_val, y_churn_train, y_churn_val, y_clv_train, y_clv_val, y_next_train, y_next_val = train_test_split(
        X_train_val, y_churn_train_val, y_clv_train_val, y_next_train_val, test_size=(15/85), random_state=42, stratify=y_churn_train_val
    )
    
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_val": X_train_val,
        "y_churn_train": y_churn_train, "y_churn_val": y_churn_val, "y_churn_test": y_churn_test,
        "y_churn_train_val": y_churn_train_val,
        "y_clv_train": y_clv_train, "y_clv_val": y_clv_val, "y_clv_test": y_clv_test,
        "y_clv_train_val": y_clv_train_val,
        "y_next_train": y_next_train, "y_next_val": y_next_val, "y_next_test": y_next_test,
        "y_next_train_val": y_next_train_val
    }
    
    logger.info(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]} | Test size: {X_test.shape[0]}")
    return splits

def preprocess_features(splits):
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    X_train_scaled = pd.DataFrame(scaler.fit_transform(splits["X_train"]), columns=splits["X_train"].columns)
    X_val_scaled = pd.DataFrame(scaler.transform(splits["X_val"]), columns=splits["X_val"].columns)
    X_test_scaled = pd.DataFrame(scaler.transform(splits["X_test"]), columns=splits["X_test"].columns)
    X_train_val_scaled = pd.DataFrame(scaler.transform(splits["X_train_val"]), columns=splits["X_train_val"].columns)
    
    splits_scaled = splits.copy()
    splits_scaled["X_train"] = X_train_scaled
    splits_scaled["X_val"] = X_val_scaled
    splits_scaled["X_test"] = X_test_scaled
    splits_scaled["X_train_val"] = X_train_val_scaled
    
    return splits_scaled, scaler

def build_churn_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.AUC(name='auc')])
    return model

def build_clv_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dense(1, activation='linear')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')])
    return model

def build_next_category_model(input_dim, num_classes):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def train_nn_model(model, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stopping],
        verbose=0
    )
    return history

def run_churn_cv(X_train_val, y_churn_train_val):
    logger.info("Running 5-Fold Stratified Cross-Validation on Churn Classification...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    auc_scores = []
    recall_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_churn_train_val)):
        X_tr, X_va = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_va = y_churn_train_val.iloc[train_idx], y_churn_train_val.iloc[val_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        model = build_churn_model(X_tr_sc.shape[1])
        train_nn_model(model, X_tr_sc, y_tr, X_va_sc, y_va)
        
        probs = model.predict(X_va_sc, verbose=0).flatten()
        preds = (probs >= 0.5).astype(int)
        
        auc_scores.append(roc_auc_score(y_va, probs))
        recall_scores.append(recall_score(y_va, preds, zero_division=0))
        
    mean_auc = np.mean(auc_scores)
    std_auc = np.std(auc_scores)
    mean_recall = np.mean(recall_scores)
    std_recall = np.std(recall_scores)
    
    logger.info(f"Churn NN 5-Fold CV Results -> ROC-AUC: {mean_auc:.4f} (+/- {std_auc:.4f}) | Recall: {mean_recall:.4f} (+/- {std_recall:.4f})")
    return mean_auc, mean_recall

def run_clv_cv(X_train_val, y_clv_train_val, y_churn_train_val):
    logger.info("Running 5-Fold Stratified Cross-Validation on CLV Regression...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    r2_scores = []
    rmse_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_churn_train_val)):
        X_tr, X_va = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_va = y_clv_train_val.iloc[train_idx], y_clv_train_val.iloc[val_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        model = build_clv_model(X_tr_sc.shape[1])
        train_nn_model(model, X_tr_sc, y_tr, X_va_sc, y_va)
        
        preds = model.predict(X_va_sc, verbose=0).flatten()
        preds = np.maximum(0, preds)
        
        r2_scores.append(r2_score(y_va, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_va, preds)))
        
    mean_r2 = np.mean(r2_scores)
    std_r2 = np.std(r2_scores)
    mean_rmse = np.mean(rmse_scores)
    std_rmse = np.std(rmse_scores)
    
    logger.info(f"CLV NN 5-Fold CV Results -> R2: {mean_r2:.4f} (+/- {std_r2:.4f}) | RMSE: {mean_rmse:.4f} (+/- {std_rmse:.4f})")
    return mean_r2, mean_rmse

def run_next_category_cv(X_train_val, y_next_train_val, y_churn_train_val, num_classes):
    logger.info("Running 5-Fold Stratified Cross-Validation on Next Category Classification...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    acc_scores = []
    f1_scores = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_churn_train_val)):
        X_tr, X_va = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_va = y_next_train_val.iloc[train_idx], y_next_train_val.iloc[val_idx]
        
        scaler = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_va_sc = scaler.transform(X_va)
        
        model = build_next_category_model(X_tr_sc.shape[1], num_classes)
        train_nn_model(model, X_tr_sc, y_tr, X_va_sc, y_va)
        
        probs = model.predict(X_va_sc, verbose=0)
        preds = np.argmax(probs, axis=1)
        
        acc_scores.append(accuracy_score(y_va, preds))
        f1_scores.append(f1_score(y_va, preds, average='weighted', zero_division=0))
        
    mean_acc = np.mean(acc_scores)
    std_acc = np.std(acc_scores)
    mean_f1 = np.mean(f1_scores)
    std_f1 = np.std(f1_scores)
    
    logger.info(f"Next Category NN 5-Fold CV Results -> Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f}) | F1-Score (Weighted): {mean_f1:.4f} (+/- {std_f1:.4f})")
    return mean_acc, mean_f1

def evaluate_classification(y_true, y_pred, y_prob=None):
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1-Score": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_prob is not None:
        metrics["ROC-AUC"] = roc_auc_score(y_true, y_prob)
    else:
        metrics["ROC-AUC"] = np.nan
    metrics["Confusion Matrix"] = confusion_matrix(y_true, y_pred).tolist()
    return metrics

def evaluate_regression(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, "..", ".."))
    
    artifacts_dir = os.path.join(project_root, "models_artifacts")
    os.makedirs(artifacts_dir, exist_ok=True)
    
    data_path = os.path.join(project_root, "data", "processed", "(PROC)online_retail_II.csv")
    X, targets, df = load_data(data_path)
    
    # Label encode the multiclass target
    logger.info("Encoding next purchase category labels...")
    label_encoder = LabelEncoder()
    y_next_encoded = pd.Series(label_encoder.fit_transform(targets["next_category"]))
    num_classes = len(label_encoder.classes_)
    logger.info(f"Number of classes for next purchase category: {num_classes}")
    
    # Save the label encoder
    joblib.dump(label_encoder, os.path.join(artifacts_dir, "label_encoder.joblib"))
    logger.info(f"Saved label encoder to {os.path.join(artifacts_dir, 'label_encoder.joblib')}")
    
    # Train / Val / Test split
    splits = split_data(X, targets["churn"], targets["clv"], y_next_encoded)
    
    # Scale features
    splits_scaled, scaler = preprocess_features(splits)
    
    # Save the scaler
    joblib.dump(scaler, os.path.join(artifacts_dir, "dl_scaler.joblib"))
    logger.info(f"Saved scaler to {os.path.join(artifacts_dir, 'dl_scaler.joblib')}")
    
    # --- CHURN CLASSIFICATION ---
    run_churn_cv(splits_scaled["X_train_val"], splits_scaled["y_churn_train_val"])
    
    logger.info("Training final Churn NN model...")
    churn_model = build_churn_model(splits_scaled["X_train"].shape[1])
    train_nn_model(churn_model, splits_scaled["X_train"], splits_scaled["y_churn_train"], splits_scaled["X_val"], splits_scaled["y_churn_val"])
    
    # Evaluate Churn NN on Test Set
    test_churn_probs = churn_model.predict(splits_scaled["X_test"]).flatten()
    test_churn_preds = (test_churn_probs >= 0.5).astype(int)
    test_churn_metrics = evaluate_classification(splits_scaled["y_churn_test"], test_churn_preds, test_churn_probs)
    
    logger.info(f"\n==========================================")
    logger.info(f"CHURN DL MODEL ON TEST SET:")
    for k, v in test_churn_metrics.items():
        if k != "Confusion Matrix":
            logger.info(f"{k}: {v:.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(test_churn_metrics['Confusion Matrix'])}")
    logger.info(f"==========================================\n")
    
    # Save best churn model
    churn_model.save(os.path.join(artifacts_dir, "best_dl_churn_model.keras"))
    logger.info(f"Saved best churn model to {os.path.join(artifacts_dir, 'best_dl_churn_model.keras')}")
    
    # --- CLV REGRESSION ---
    run_clv_cv(splits_scaled["X_train_val"], splits_scaled["y_clv_train_val"], splits_scaled["y_churn_train_val"])
    
    logger.info("Training final CLV NN model...")
    clv_model = build_clv_model(splits_scaled["X_train"].shape[1])
    train_nn_model(clv_model, splits_scaled["X_train"], splits_scaled["y_clv_train"], splits_scaled["X_val"], splits_scaled["y_clv_val"])
    
    # Evaluate CLV NN on Test Set
    test_clv_preds = clv_model.predict(splits_scaled["X_test"]).flatten()
    test_clv_preds = np.maximum(0, test_clv_preds)
    test_clv_metrics = evaluate_regression(splits_scaled["y_clv_test"], test_clv_preds)
    
    logger.info(f"\n==========================================")
    logger.info(f"CLV DL MODEL ON TEST SET:")
    for k, v in test_clv_metrics.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info(f"==========================================\n")
    
    # Save best CLV model
    clv_model.save(os.path.join(artifacts_dir, "best_dl_clv_model.keras"))
    logger.info(f"Saved best CLV model to {os.path.join(artifacts_dir, 'best_dl_clv_model.keras')}")
    
    # --- NEXT CATEGORY MULTICLASS CLASSIFICATION ---
    run_next_category_cv(splits_scaled["X_train_val"], splits_scaled["y_next_train_val"], splits_scaled["y_churn_train_val"], num_classes)
    
    logger.info("Training final Next Category NN model...")
    next_category_model = build_next_category_model(splits_scaled["X_train"].shape[1], num_classes)
    train_nn_model(next_category_model, splits_scaled["X_train"], splits_scaled["y_next_train"], splits_scaled["X_val"], splits_scaled["y_next_val"])
    
    # Evaluate Next Category NN on Test Set
    test_next_probs = next_category_model.predict(splits_scaled["X_test"])
    test_next_preds = np.argmax(test_next_probs, axis=1)
    
    # Calculate weighted multiclass metrics
    next_accuracy = accuracy_score(splits_scaled["y_next_test"], test_next_preds)
    next_precision = precision_score(splits_scaled["y_next_test"], test_next_preds, average='weighted', zero_division=0)
    next_recall = recall_score(splits_scaled["y_next_test"], test_next_preds, average='weighted', zero_division=0)
    next_f1 = f1_score(splits_scaled["y_next_test"], test_next_preds, average='weighted', zero_division=0)
    next_conf_matrix = confusion_matrix(splits_scaled["y_next_test"], test_next_preds)
    
    logger.info(f"\n==========================================")
    logger.info(f"NEXT CATEGORY DL MODEL ON TEST SET (WEIGHTED METRICS):")
    logger.info(f"Accuracy: {next_accuracy:.4f}")
    logger.info(f"Precision: {next_precision:.4f}")
    logger.info(f"Recall: {next_recall:.4f}")
    logger.info(f"F1-Score: {next_f1:.4f}")
    logger.info(f"Confusion Matrix shape: {next_conf_matrix.shape}")
    logger.info(f"==========================================\n")
    
    # Save best next category model
    next_category_model.save(os.path.join(artifacts_dir, "best_dl_next_category_model.keras"))
    logger.info(f"Saved best next category model to {os.path.join(artifacts_dir, 'best_dl_next_category_model.keras')}")

if __name__ == "__main__":
    main()
