import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Lasso

from sklearn.metrics import (mean_squared_error, r2_score, accuracy_score,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)

import joblib
import os


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from CSV file
    
    Parameters:
    file_path (str): Path to CSV file
    
    Returns:
    pd.DataFrame: Loaded dataframe
    """
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    return df


def convert_income(income_str):
    if pd.isna(income_str):
        return np.nan
    
    elif "75.000-" in income_str:
        return 75.0
    
    elif "50.000-75.000" in income_str:
        return 62.5
    
    elif "40.000-50.000" in income_str:
        return 45.0
    
    elif "30.000-40.000" in income_str:
        return 35.0
    
    elif "25.000-30.000" in income_str:
        return 27.5
    
    elif "20.000-25.000" in income_str:
        return 22.5
    
    elif "15.000-20.000" in income_str:
        return 17.5
    
    elif "10.000-15.000" in income_str:
        return 12.5
    
    elif "-10.000" in income_str:
        return 5.0

    else:
        return np.nan


def convert_age(age_str):
    if pd.isna(age_str):
        return np.nan
    
    elif "-" in age_str:
        low, high = age_str.split("-")
        return (float(low) + float(high)) / 2
    
    elif "+" in age_str:
        return float(age_str.replace("+", ""))
    
    else:
        return np.nan


def drop_transform_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop and transform columns in dataset
    
    Parameters:
    df (pd.DataFrame): Input dataframe
    
    Returns
    """
    # Drop columns
    df = df.drop(['UNDER18', 'AREA', 'HOME.TYPE'], axis=1)
    
    # Transform columns
    df["INCOME"] = df["INCOME"].apply(convert_income)

    size_map = {
    "None": 0, "One": 1, "Two": 2, "Three": 3, "Four": 4,
    "Five": 5, "Six": 6, "Seven": 7, "Eight": 8, "Nine or more": 9
    }
    df["HOUSEHOLD.SIZE"] = df["HOUSEHOLD.SIZE"].map(size_map)

    df["AGE"] = df["AGE"].apply(convert_age)

    # Convertir DUAL.INCOMES a binario
    df["DUAL.INCOMES"] = df["DUAL.INCOMES"].map({"Yes": 1, "No": 0})

    # Convertir EDUCATION a ordinal
    education_order = {
    "Grade 8 or less": 1,
    "Grades 9 to 11": 2,
    "Graduated High School": 3,
    "1 to 3 years of college": 4,
    "College graduate": 5,
    "Grad Study": 6
    }

    df["EDUCATION"] = df["EDUCATION"].map(education_order)
    


    return df


def show_stats(df: pd.DataFrame):
    """
    Show basic statistics and target distribution
    """
    print(df.info())

    sns.histplot(df['INCOME'])
    plt.show()


def build_preprocessor() -> ColumnTransformer:
    """
    Create preprocessing pipeline for numerical and categorical features
    
    Returns:
    ColumnTransformer: Configured preprocessing pipeline
    """
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), NUM_FEATURES),
            ('cat', Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ]), CAT_FEATURES)
        ])



def save_model(model, file_path: str):
    """
    Save trained model to disk
    
    Parameters:
    model: Trained model object
    file_path (str): Path to save model
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    joblib.dump(model, file_path)
    print(f"Model saved to {file_path}")



def model_pipeline(df: pd.DataFrame):
    """
    Main function to train the regression pipeline
    and save the best model
    
    Steps:
    1. Split into features/target
    2. Create complete pipeline
    3. Perform grid search
    4. Save best model

    Returns:
    GridSearchCV: Best model found
    """
    # Split features and target
    X = df.drop('INCOME', axis=1)
    y = df['INCOME']
    
    # Define models to compare
    models = {
        'random_forest': RandomForestRegressor(),
        'gradient_boosting': GradientBoostingRegressor(),
        'Lasso': Lasso()
    }

    best_score = np.inf
    best_model = None
    
    # Compare models
    for name, model in models.items():
        pipeline = Pipeline(steps=[
            ('preprocessor', build_preprocessor()),
            ('regressor', model)
        ])
        
        # Basic cross-validation
        scores = -cross_val_score(pipeline, X, y, 
                                cv=3, scoring='neg_mean_squared_error')
        avg_mse = np.mean(scores)
        
        print(f"{name} - Average MSE: {avg_mse:.2f}")
        
        if avg_mse < best_score:
            best_score = avg_mse
            best_model = name

    print(f"\nBest model: {best_model} with MSE: {best_score:.2f}")

    # Set up GridSearchCV for best model
    param_grids = {
        'random_forest': {
            'regressor__n_estimators': [100, 200],
            'regressor__max_depth': [None, 10, 20]
        },
        'gradient_boosting': {
            'regressor__n_estimators': [100, 200],
            'regressor__learning_rate': [0.1, 0.05]
        },
        'svr': {
            'regressor__C': [1, 10],
            'regressor__epsilon': [0.1, 0.2]
        }
    }

    final_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('regressor', models[best_model])
    ])

    grid_search = GridSearchCV(
        estimator=final_pipeline,
        param_grid=param_grids[best_model],
        cv=5,
        scoring='neg_mean_squared_error',
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    # Save best model
    save_model(grid_search.best_estimator_, '../models/best_regression_pipeline_model.pkl')
    
    return grid_search

# Define numerical and categorical features 
NUM_FEATURES = ['AGE', 'EDUCATION', 'DUAL.INCOMES', "HOUSEHOLD.SIZE"]
CAT_FEATURES = ['SEX', 'LANGUAGE', "ETHNIC.CLASS",
                            "HOUSEHOLDER","MARITAL.STATUS", "OCCUPATION"]



# CASO ESPECIAL DE CLASIFICACIÓN DESBALANCEADA
from sklearn.ensemble import RandomForestClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

def imbalanced_classification_pipeline():
    """
    Pipeline para clasificación desbalanceada
    """
    # Load data
    train_df = load_data('../data/train.csv')
    X = train_df.drop('rare_class', axis=1)
    y = train_df['rare_class']  # Ejemplo: clase minoritaria <5%
    
    # Pipeline con balanceo
    pipeline = ImbPipeline(steps=[
        ('preprocessor', build_preprocessor()),  # Preprocesamiento
        ('smote', SMOTE(sampling_strategy=0.3)),  # Sobremuestreo
        ('classifier', RandomForestClassifier(class_weight='balanced'))
    ])
    
    # Parámetros y métricas
    param_grid = {
        'classifier__n_estimators': [100, 200],
        'smote__k_neighbors': [3, 5]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid,
                             scoring='roc_auc',
                             cv=5, n_jobs=-1)
    
    grid_search.fit(X, y)
    
    # Evaluación especial
    y_pred = grid_search.predict(X)
    print("Balanced Accuracy:", balanced_accuracy_score(y, y_pred))
    print("ROC AUC:", roc_auc_score(y, y_pred))
    
    save_model(grid_search.best_estimator_, '../models/imbalanced_pipeline.pkl')
    return grid_search


# CASO ESPECIAL DE NO GAUSIANA (POISSON)
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import PowerTransformer

def non_gaussian_pipeline():
    """
    Pipeline for targets with non-Gaussian distributions
    """
    # Load data
    train_df = load_data('../data/train.csv')
    X = train_df.drop('count_target', axis=1)
    y = train_df['count_target']  # Ejemplo: conteos (Poisson distribution)
    
    # Pipeline especial
    pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),  # Preprocesamiento
        ('power_transform', PowerTransformer(method='yeo-johnson')),  # Transformación
        ('regressor', PoissonRegressor(max_iter=1000))
    ])
    
    # Parámetros para GridSearch
    param_grid = {
        'regressor__alpha': [0.1, 1.0],
        'power_transform__standardize': [True, False]
    }
    
    grid_search = GridSearchCV(pipeline, param_grid, 
                             scoring='neg_mean_poisson_deviance',
                             cv=5, n_jobs=-1)
    
    grid_search.fit(X, y)
    
    print("Best Poisson Regression Params:", grid_search.best_params_)
    save_model(grid_search.best_estimator_, '../models/poisson_pipeline_model.pkl')
    
    return grid_search



### FUNCIONES DEL ARCHIVO PIPELINES_II ###

def load_model(file_path: str):
    """
    Load trained pipeline from disk
    
    Parameters:
    file_path (str): Path to saved model
    
    Returns:
    Pipeline: Loaded sklearn pipeline
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No model found at {file_path}")
        
    model = joblib.load(file_path)
    print(f"Successfully loaded model from {file_path}")
    return model



def make_predictions(model, test_df: pd.DataFrame) -> np.ndarray:
    """
    Generate predictions using trained pipeline
    
    Parameters:
    model: Trained sklearn pipeline
    test_df (pd.DataFrame): Test dataset
    
    Returns:
    np.ndarray: Array of predictions
    """
    predictions = model.predict(test_df)
    print(f"Generated {len(predictions)} predictions")
    return predictions


def evaluate_model(y_true: pd.Series, y_pred: np.ndarray):
    """
    Calculate evaluation metrics and generate visualizations
    
    Parameters:
    y_true (pd.Series): True target values
    y_pred (np.ndarray): Model predictions
    proba_pred (np.ndarray): Predicted probabilities (for classification)
    """
    # Regression Metrics
    print("Regression Metrics:")
    print(f"MSE: {mean_squared_error(y_true, y_pred):.3f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_true, y_pred)):.3f}")
    print(f"R²: {r2_score(y_true, y_pred):.3f}")
    
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion Matrix Visualization
    plt.figure(figsize=(8,6))
    sns.heatmap(confusion_matrix(y_true, y_pred), 
                annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    plt.show()


def save_results(y_pred: np.ndarray, file_path: str):
    """
    Save predictions to CSV file
    
    Parameters:
    y_pred (np.ndarray): Array of predictions
    file_path (str): Path to save predictions
    """
    results_df = pd.DataFrame(y_pred, columns=['predictions'])
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    results_df.to_csv(file_path, index=False)
    print(f"Predictions saved to {file_path}")