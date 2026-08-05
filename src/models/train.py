import os
import logging
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
    mean_squared_error, r2_score, mean_absolute_error
)
from src.utils.eda_utils import calculate_vif

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(os.path.dirname(os.path.abspath(__file__)), "training.log"), mode="w")
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

def split_data(X, y_churn, y_clv):
    logger.info("Splitting dataset into train (70%), validation (15%), and test (15%)...")
    
    # Split 85% train_val and 15% test
    X_train_val, X_test, y_churn_train_val, y_churn_test, y_clv_train_val, y_clv_test = train_test_split(
        X, y_churn, y_clv, test_size=0.15, random_state=42, stratify=y_churn
    )
    
    # Split train_val (85%) into train (70% total) and validation (15% total)
    # Ratio: 15 / 85 = 0.17647
    X_train, X_val, y_churn_train, y_churn_val, y_clv_train, y_clv_val = train_test_split(
        X_train_val, y_churn_train_val, y_clv_train_val, test_size=(15/85), random_state=42, stratify=y_churn_train_val
    )
    
    splits = {
        "X_train": X_train, "X_val": X_val, "X_test": X_test,
        "X_train_val": X_train_val,
        "y_churn_train": y_churn_train, "y_churn_val": y_churn_val, "y_churn_test": y_churn_test,
        "y_churn_train_val": y_churn_train_val,
        "y_clv_train": y_clv_train, "y_clv_val": y_clv_val, "y_clv_test": y_clv_test,
        "y_clv_train_val": y_clv_train_val
    }
    
    logger.info(f"Train size: {X_train.shape[0]} | Val size: {X_val.shape[0]} | Test size: {X_test.shape[0]}")
    return splits

def preprocess_features(splits):
    logger.info("Scaling features using StandardScaler...")
    scaler = StandardScaler()
    
    # Fit on training set only
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

def run_vif_analysis(X):
    logger.info("Running VIF analysis to test Hypothesis 4 (Multicollinearity)...")
    continuous_cols = [
        "recency", "frequency", "monetary_total", "monetary_avg", 
        "avg_basket_size", "purchase_frequency_trend", 
        "variance_purchase_intervals", "seasonal_concentration", 
        "return_rate", "discount_sensitivity", "historical_clv", 
        "engagement_score"
    ]
    
    vif_with = calculate_vif(X, continuous_cols)
    vif_without = calculate_vif(X, [col for col in continuous_cols if col != "engagement_score"])
    
    logger.info("\n--- VIF WITH engagement_score ---\n" + vif_with.to_string(index=False))
    logger.info("\n--- VIF WITHOUT engagement_score ---\n" + vif_without.to_string(index=False))
    
    return vif_with, vif_without

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

def train_churn_classifiers(splits_scaled):
    logger.info("Training Churn Classification models...")
    X_train = splits_scaled["X_train"]
    y_train = splits_scaled["y_churn_train"]
    X_val = splits_scaled["X_val"]
    y_val = splits_scaled["y_churn_val"]
    
    # 1. Logistic Regression (with engagement_score)
    lr_with = LogisticRegression(max_iter=1000, random_state=42, penalty='l2')
    lr_with.fit(X_train, y_train)
    
    # 2. Logistic Regression (without engagement_score)
    X_train_no_eng = X_train.drop(columns=["engagement_score"])
    X_val_no_eng = X_val.drop(columns=["engagement_score"])
    lr_without = LogisticRegression(max_iter=1000, random_state=42, penalty='l2')
    lr_without.fit(X_train_no_eng, y_train)
    
    # Evaluate Hypothesis 4: instability of coefficients
    coef_df = pd.DataFrame({
        "feature": [col for col in X_train.columns if col != "engagement_score"],
        "coef_with_eng": [lr_with.coef_[0][i] for i, col in enumerate(X_train.columns) if col != "engagement_score"],
        "coef_without_eng": lr_without.coef_[0]
    })
    logger.info("\n--- Hypothesis 4: Coefficient Comparison ---\n" + coef_df.to_string(index=False))
    
    # 3. Random Forest Classifier
    rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', max_depth=10)
    rf.fit(X_train, y_train)
    
    # 4. Gradient Boosting Classifier
    gb = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=4)
    gb.fit(X_train, y_train)
    
    models = {
        "Logistic Regression (With Eng)": (lr_with, X_train, X_val),
        "Logistic Regression (Without Eng)": (lr_without, X_train_no_eng, X_val_no_eng),
        "Random Forest": (rf, X_train, X_val),
        "Gradient Boosting": (gb, X_train, X_val)
    }
    
    results = {}
    for name, (model, xtr, xval) in models.items():
        preds = model.predict(xval)
        probs = model.predict_proba(xval)[:, 1]
        metrics = evaluate_classification(y_val, preds, probs)
        results[name] = metrics
        logger.info(f"--- {name} (Validation Set) ---")
        for k, v in metrics.items():
            if k != "Confusion Matrix":
                logger.info(f"{k}: {v:.4f}")
                
    return models, results, coef_df

def run_churn_cross_validation(X_train_val_scaled, y_train_val):
    logger.info("Running 5-Fold Stratified Cross-Validation on Churn Classification...")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models_to_test = {
        "Logistic Regression (With Eng)": lambda: LogisticRegression(max_iter=1000, random_state=42, penalty='l2'),
        "Random Forest": lambda: RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', max_depth=10),
        "Gradient Boosting": lambda: GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=4)
    }
    
    cv_results = {}
    
    for name, model_fn in models_to_test.items():
        auc_scores = []
        recall_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_scaled, y_train_val)):
            X_tr, X_va = X_train_val_scaled.iloc[train_idx], X_train_val_scaled.iloc[val_idx]
            y_tr, y_va = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
            
            model = model_fn()
            model.fit(X_tr, y_tr)
            
            preds = model.predict(X_va)
            probs = model.predict_proba(X_va)[:, 1]
            
            auc_scores.append(roc_auc_score(y_va, probs))
            recall_scores.append(recall_score(y_va, preds, zero_division=0))
            
        cv_results[name] = {
            "Mean ROC-AUC": np.mean(auc_scores),
            "Std ROC-AUC": np.std(auc_scores),
            "Mean Recall": np.mean(recall_scores),
            "Std Recall": np.std(recall_scores)
        }
        logger.info(f"{name} 5-Fold CV Results -> "
                    f"ROC-AUC: {cv_results[name]['Mean ROC-AUC']:.4f} (+/- {cv_results[name]['Std ROC-AUC']:.4f}) | "
                    f"Recall: {cv_results[name]['Mean Recall']:.4f} (+/- {cv_results[name]['Std Recall']:.4f})")
        
    return cv_results

def evaluate_regression(y_true, y_pred):
    return {
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
        "MAE": mean_absolute_error(y_true, y_pred)
    }

def train_clv_regressors(splits):
    logger.info("Training CLV Regression models...")
    X_train = splits["X_train"]
    y_train = splits["y_clv_train"]
    X_val = splits["X_val"]
    y_val = splits["y_clv_val"]
    
    # Scaler for regression model inputs (linear models benefit from it)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_val_scaled = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    
    # 1. Ridge Regressor (Raw features -> Raw target)
    ridge_raw = Ridge(alpha=1.0)
    ridge_raw.fit(X_train_scaled, y_train)
    preds_ridge_raw = ridge_raw.predict(X_val_scaled)
    metrics_ridge_raw = evaluate_regression(y_val, preds_ridge_raw)
    
    # 2. Ridge Regressor (Log-transformed features -> Log-transformed target) - Hypothesis 3
    # Skewed features to transform
    skewed_cols = [
        "recency", "frequency", "monetary_total", "monetary_avg", 
        "avg_basket_size", "variance_purchase_intervals", "historical_clv"
    ]
    
    def log_transform_df(df):
        df_log = df.copy()
        for col in skewed_cols:
            df_log[col] = np.log1p(np.maximum(0, df_log[col]))
        return df_log
        
    X_train_log = log_transform_df(X_train)
    X_val_log = log_transform_df(X_val)
    
    scaler_log = StandardScaler()
    X_train_log_scaled = pd.DataFrame(scaler_log.fit_transform(X_train_log), columns=X_train_log.columns)
    X_val_log_scaled = pd.DataFrame(scaler_log.transform(X_val_log), columns=X_val_log.columns)
    
    y_train_log = np.log1p(np.maximum(0, y_train))
    
    ridge_log = Ridge(alpha=1.0)
    ridge_log.fit(X_train_log_scaled, y_train_log)
    
    preds_ridge_log_scale = ridge_log.predict(X_val_log_scaled)
    # Convert back to raw scale
    preds_ridge_log_raw = np.expm1(preds_ridge_log_scale)
    # Clip negative predictions to 0
    preds_ridge_log_raw = np.maximum(0, preds_ridge_log_raw)
    metrics_ridge_log = evaluate_regression(y_val, preds_ridge_log_raw)
    
    logger.info(f"\n--- Hypothesis 3: CLV Regression on Raw vs Log-transformed ---")
    logger.info(f"Raw Target Model    -> R2: {metrics_ridge_raw['R2']:.4f} | RMSE: {metrics_ridge_raw['RMSE']:.4f} | MAE: {metrics_ridge_raw['MAE']:.4f}")
    logger.info(f"Log Target Model    -> R2: {metrics_ridge_log['R2']:.4f} | RMSE: {metrics_ridge_log['RMSE']:.4f} | MAE: {metrics_ridge_log['MAE']:.4f}")
    
    # 3. Random Forest Regressor
    rf_reg = RandomForestRegressor(random_state=42, n_estimators=100, max_depth=10)
    rf_reg.fit(X_train, y_train)
    preds_rf = rf_reg.predict(X_val)
    metrics_rf = evaluate_regression(y_val, preds_rf)
    
    # 4. Gradient Boosting Regressor
    gb_reg = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=4)
    gb_reg.fit(X_train, y_train)
    preds_gb = gb_reg.predict(X_val)
    metrics_gb = evaluate_regression(y_val, preds_gb)
    
    models = {
        "Ridge (Raw)": (ridge_raw, X_train_scaled, X_val_scaled),
        "Ridge (Log)": (ridge_log, X_train_log_scaled, X_val_log_scaled, scaler_log, log_transform_df),
        "Random Forest Regressor": (rf_reg, X_train, X_val),
        "Gradient Boosting Regressor": (gb_reg, X_train, X_val)
    }
    
    results = {
        "Ridge (Raw)": metrics_ridge_raw,
        "Ridge (Log)": metrics_ridge_log,
        "Random Forest Regressor": metrics_rf,
        "Gradient Boosting Regressor": metrics_gb
    }
    
    for name, metrics in results.items():
        logger.info(f"--- {name} (Validation Set) ---")
        for k, v in metrics.items():
            logger.info(f"{k}: {v:.4f}")
            
    return models, results

def run_clv_cross_validation(X_train_val, y_train_val):
    logger.info("Running 5-Fold Cross-Validation on CLV Regression...")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # We will test Gradient Boosting and Random Forest Regressors, and Ridge (Log)
    skewed_cols = [
        "recency", "frequency", "monetary_total", "monetary_avg", 
        "avg_basket_size", "variance_purchase_intervals", "historical_clv"
    ]
    def log_transform_df(df):
        df_log = df.copy()
        for col in skewed_cols:
            df_log[col] = np.log1p(np.maximum(0, df_log[col]))
        return df_log
        
    cv_results = {}
    
    # 1. Gradient Boosting Regressor
    r2_scores = []
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train_val):
        X_tr, X_va = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_va = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        model = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=4)
        model.fit(X_tr, y_tr)
        preds = model.predict(X_va)
        
        r2_scores.append(r2_score(y_va, preds))
        rmse_scores.append(np.sqrt(mean_squared_error(y_va, preds)))
    cv_results["Gradient Boosting Regressor"] = {
        "Mean R2": np.mean(r2_scores), "Std R2": np.std(r2_scores),
        "Mean RMSE": np.mean(rmse_scores), "Std RMSE": np.std(rmse_scores)
    }
    
    # 2. Ridge (Log)
    r2_scores = []
    rmse_scores = []
    for train_idx, val_idx in kf.split(X_train_val):
        X_tr, X_va = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
        y_tr, y_va = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]
        
        X_tr_log = log_transform_df(X_tr)
        X_va_log = log_transform_df(X_va)
        
        scaler = StandardScaler()
        X_tr_log_sc = scaler.fit_transform(X_tr_log)
        X_va_log_sc = scaler.transform(X_va_log)
        
        y_tr_log = np.log1p(np.maximum(0, y_tr))
        
        model = Ridge(alpha=1.0)
        model.fit(X_tr_log_sc, y_tr_log)
        
        preds_log = model.predict(X_va_log_sc)
        preds_raw = np.maximum(0, np.expm1(preds_log))
        
        r2_scores.append(r2_score(y_va, preds_raw))
        rmse_scores.append(np.sqrt(mean_squared_error(y_va, preds_raw)))
    cv_results["Ridge (Log)"] = {
        "Mean R2": np.mean(r2_scores), "Std R2": np.std(r2_scores),
        "Mean RMSE": np.mean(rmse_scores), "Std RMSE": np.std(rmse_scores)
    }
    
    for name, res in cv_results.items():
        logger.info(f"{name} 5-Fold CV Results -> "
                    f"R2: {res['Mean R2']:.4f} (+/- {res['Std R2']:.4f}) | "
                    f"RMSE: {res['Mean RMSE']:.4f} (+/- {res['Std RMSE']:.4f})")
        
    return cv_results

def main():
    from pathlib import Path
    Path("models_artifacts").mkdir(parents=True, exist_ok=True)
    
    data_path = "data/processed/(PROC)online_retail_II.csv"
    X, targets, df = load_data(data_path)
    
    # VIF Analysis
    run_vif_analysis(X)
    
    # Train / Val / Test split
    splits = split_data(X, targets["churn"], targets["clv"])
    
    # Scale features
    splits_scaled, scaler = preprocess_features(splits)
    
    # Save the scaler
    joblib.dump(scaler, "models_artifacts/scaler.joblib")
    logger.info("Saved scaler to models_artifacts/scaler.joblib")
    
    # --- CHURN CLASSIFICATION ---
    churn_models, churn_results, coef_df = train_churn_classifiers(splits_scaled)
    run_churn_cross_validation(splits_scaled["X_train_val"], splits_scaled["y_churn_train_val"])
    
    # Choose best churn model based on Val ROC-AUC
    best_churn_name = "Gradient Boosting"
    best_churn_model = churn_models[best_churn_name][0]
    
    # Evaluate best churn model on Test Set
    X_test_scaled = splits_scaled["X_test"]
    y_test_churn = splits_scaled["y_churn_test"]
    test_churn_preds = best_churn_model.predict(X_test_scaled)
    test_churn_probs = best_churn_model.predict_proba(X_test_scaled)[:, 1]
    test_churn_metrics = evaluate_classification(y_test_churn, test_churn_preds, test_churn_probs)
    
    logger.info(f"\n==========================================")
    logger.info(f"BEST CHURN MODEL ON TEST SET: {best_churn_name}")
    for k, v in test_churn_metrics.items():
        if k != "Confusion Matrix":
            logger.info(f"{k}: {v:.4f}")
    logger.info(f"Confusion Matrix:\n{np.array(test_churn_metrics['Confusion Matrix'])}")
    logger.info(f"==========================================\n")
    
    # Save best churn model
    joblib.dump(best_churn_model, "models_artifacts/best_churn_model.joblib")
    logger.info("Saved best churn model to models_artifacts/best_churn_model.joblib")
    
    # --- CLV REGRESSION ---
    clv_models, clv_results = train_clv_regressors(splits)
    run_clv_cross_validation(splits["X_train_val"], splits["y_clv_train_val"])
    
    # Choose best CLV model based on Val R2
    best_clv_name = "Gradient Boosting Regressor"
    best_clv_model = clv_models[best_clv_name][0]
    
    # Evaluate best CLV model on Test Set
    X_test = splits["X_test"]
    y_test_clv = splits["y_clv_test"]
    test_clv_preds = best_clv_model.predict(X_test)
    test_clv_metrics = evaluate_regression(y_test_clv, test_clv_preds)
    
    logger.info(f"\n==========================================")
    logger.info(f"BEST CLV MODEL ON TEST SET: {best_clv_name}")
    for k, v in test_clv_metrics.items():
        logger.info(f"{k}: {v:.4f}")
    logger.info(f"==========================================\n")
    
    # Save best CLV model
    joblib.dump(best_clv_model, "models_artifacts/best_clv_model.joblib")
    logger.info("Saved best CLV model to models_artifacts/best_clv_model.joblib")

if __name__ == "__main__":
    main()
