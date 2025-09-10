import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np
import joblib

X = pd.read_csv('features.csv')
y = pd.read_csv('outputs.csv')

# Standardizing features due to different value ranges
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

models = {}
metrics = {}

for column in y.columns:
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y[column], test_size=0.2, random_state=42
    )
    
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    models[column] = model
    
    y_pred = model.predict(X_test)
    metrics[column] = {
        'r2': r2_score(y_test, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_test, y_pred))
    }
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'coefficient': model.coef_,
        'abs_coefficient': abs(model.coef_)
    }).sort_values('abs_coefficient', ascending=False)

    feature_importance.to_csv(f'feature_importance_{column}.csv', index=False)

    
    print(f"\nResults for {column}:")
    print(f"R² Score: {metrics[column]['r2']:.3f}")
    print(f"RMSE: {metrics[column]['rmse']:.3f}")
    #print("\nTop 5 features:")
    #print(feature_importance.head())

    joblib.dump(model, f'model_{column}.joblib')
    
joblib.dump(scaler, f'scaler.joblib')

# To load later:
# loaded_model = joblib.load('model_columnname.joblib')
# loaded_scaler = joblib.load('scaler.joblib')