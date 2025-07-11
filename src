import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score

def load_and_prepare_data():
    """Load and prepare the Boston Housing dataset."""
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]

    columns = [
        "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE",
        "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT", "PRICE"
    ]
    df = pd.DataFrame(np.column_stack([data, target]), columns=columns)

    # Split data into features and target
    X = df.drop('PRICE', axis=1)
    y = df['PRICE']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return X_train, X_test, y_train, y_test, df

def create_correlation_heatmap(data):
    """Create and display a correlation heatmap."""
    plt.figure(figsize=(10, 8))
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
    plt.title("Correlation Heatmap")
    plt.show()

def train_models(X_train, y_train):
    """Train multiple models and return them as a dictionary."""
    models = {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
    }
    for name, model in models.items():
        model.fit(X_train, y_train)
    return models

def evaluate_models(models, X_train, X_test, y_train, y_test):
    """Evaluate models and return performance metrics."""
    metrics = {}
    for name, model in models.items():
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        metrics[name] = {
            "Train RMSE": mean_squared_error(y_train, train_pred, squared=False),
            "Test RMSE": mean_squared_error(y_test, test_pred, squared=False),
            "Train R^2": r2_score(y_train, train_pred),
            "Test R^2": r2_score(y_test, test_pred),
        }
    return metrics

def plot_predictions(models, X_test, y_test):
    """Plot predictions of models against actual values."""
    plt.figure(figsize=(12, 8))
    for i, (name, model) in enumerate(models.items(), start=1):
        plt.subplot(2, 2, i)
        y_pred = model.predict(X_test)
        plt.scatter(y_test, y_pred, alpha=0.7)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
        plt.title(name)
        plt.xlabel("Actual Prices")
        plt.ylabel("Predicted Prices")
        plt.tight_layout()
    plt.show()

def analyze_feature_importance(models, feature_names):
    """Analyze feature importance for tree-based models."""
    feature_importance = {}
    for name, model in models.items():
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            feature_importance[name] = dict(zip(feature_names, importance))
    return feature_importance

def main():
    # Load and prepare data
    print("Loading and preparing data...")
    X_train, X_test, y_train, y_test, data = load_and_prepare_data()
    print("Data shape:", data.shape)
    
    # Train models
    print("\nTraining models...")
    models = train_models(X_train, y_train)
    
    # Evaluate models
    print("\nEvaluating models...")
    metrics = evaluate_models(models, X_train, X_test, y_train, y_test)
    print("\nModel Performance Metrics:")
    for model, metric in metrics.items():
        print(f"\n{model}:")
        for name, value in metric.items():
            print(f"{name}: {value:.4f}")
    
    # Analyze feature importance
    print("\nAnalyzing feature importance...")
    feature_importance = analyze_feature_importance(models, data.drop('PRICE', axis=1).columns)
    print("\nFeature Importance:")
    for model, importance in feature_importance.items():
        print(f"\n{model}:")
        for feature, value in sorted(importance.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {value:.4f}")
    
    # Create correlation heatmap
    print("\nCreating correlation heatmap...")
    create_correlation_heatmap(data)
      
    # Plot predictions
    print("\nPlotting predictions...")
    plot_predictions(models, X_test, y_test)

if __name__ == "__main__":
    main()
