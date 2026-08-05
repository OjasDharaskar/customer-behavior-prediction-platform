import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import roc_curve, confusion_matrix, roc_auc_score, recall_score, precision_score, accuracy_score, f1_score, mean_squared_error, r2_score, mean_absolute_error
from src.models.train import load_data, split_data, preprocess_features

def main():
    from pathlib import Path
    Path("models_artifacts").mkdir(parents=True, exist_ok=True)
    
    # Load and preprocess data
    data_path = "data/processed/(PROC)online_retail_II.csv"
    X, targets, df = load_data(data_path)
    splits = split_data(X, targets["churn"], targets["clv"])
    splits_scaled, scaler = preprocess_features(splits)
    
    X_train_scaled = splits_scaled["X_train"]
    y_train = splits_scaled["y_churn_train"]
    X_val_scaled = splits_scaled["X_val"]
    y_val = splits_scaled["y_churn_val"]
    
    # Train classifiers
    lr_with = LogisticRegression(max_iter=1000, random_state=42, penalty='l2')
    lr_with.fit(X_train_scaled, y_train)
    
    rf = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced', max_depth=10)
    rf.fit(X_train_scaled, y_train)
    
    gb = GradientBoostingClassifier(random_state=42, n_estimators=100, max_depth=4)
    gb.fit(X_train_scaled, y_train)
    
    # 1. Plot ROC Curves
    plt.figure(figsize=(10, 8))
    models = {
        "Logistic Regression": (lr_with, X_val_scaled),
        "Random Forest": (rf, X_val_scaled),
        "Gradient Boosting": (gb, X_val_scaled)
    }
    for name, (model, xval) in models.items():
        probs = model.predict_proba(xval)[:, 1]
        fpr, tpr, _ = roc_curve(y_val, probs)
        auc = roc_auc_score(y_val, probs)
        plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    
    plt.plot([0, 1], [0, 1], 'k--', label="Random Guess")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves for Churn Classification")
    plt.legend(loc="lower right")
    plt.savefig("models_artifacts/churn_roc_curve.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 2. Plot Confusion Matrix (Gradient Boosting on Test Set)
    X_test_scaled = splits_scaled["X_test"]
    y_test_churn = splits_scaled["y_churn_test"]
    test_preds = gb.predict(X_test_scaled)
    cm = confusion_matrix(y_test_churn, test_preds)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["Active (0)", "Churn (1)"], 
                yticklabels=["Active (0)", "Churn (1)"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix for Best Churn Model (Gradient Boosting on Test Set)")
    plt.savefig("models_artifacts/churn_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 3. Plot Actual vs Predicted CLV (Gradient Boosting Regressor)
    gb_reg = GradientBoostingRegressor(random_state=42, n_estimators=100, max_depth=4)
    gb_reg.fit(splits["X_train"], splits["y_clv_train"])
    
    X_test = splits["X_test"]
    y_test_clv = splits["y_clv_test"]
    test_clv_preds = gb_reg.predict(X_test)
    test_clv_preds = np.maximum(0, test_clv_preds)
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test_clv, y=test_clv_preds, alpha=0.6, color='purple')
    max_val = max(y_test_clv.max(), test_clv_preds.max())
    plt.plot([0, max_val], [0, max_val], 'r--', label="Perfect Prediction")
    plt.xlabel("Actual 90-Day CLV")
    plt.ylabel("Predicted 90-Day CLV")
    plt.title("Actual vs Predicted CLV (Best Model - Gradient Boosting Regressor)")
    plt.xscale("symlog")
    plt.yscale("symlog")
    plt.legend()
    plt.savefig("models_artifacts/clv_actual_vs_predicted.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    # 4. Feature Importances for Churn
    importances = gb.feature_importances_
    indices = np.argsort(importances)[::-1]
    features = X.columns
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x=importances[indices], y=features[indices], palette="viridis")
    plt.xlabel("Gini Importance")
    plt.ylabel("Feature")
    plt.title("Feature Importance for Best Churn Model (Gradient Boosting)")
    plt.savefig("models_artifacts/feature_importances.png", dpi=300, bbox_inches="tight")
    plt.close()
    
    print("Plots generated successfully in models_artifacts/ directory!")

if __name__ == "__main__":
    main()
