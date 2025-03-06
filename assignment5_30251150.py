import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split, RandomizedSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error
import xgboost as xgb
import matplotlib.pyplot as plt

#LOAD DATASET
diabetes = load_diabetes()
X = diabetes['data']
y = diabetes['target']
feature_names = diabetes['feature_names']
X_df = pd.DataFrame(X, columns=feature_names)
y_df = pd.DataFrame(y, columns=['target'])

#Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Parameter selection 
parameters = {
    'max_depth': np.arange(1, 6, 1),
    'learning_rate': np.arange(0.1, 1.0, 0.1),
    'n_estimators': np.arange(10, 110, 10),
    'reg_alpha': np.arange(0, 1.1, 0.1),
    'reg_lambda': np.arange(0, 1.1, 0.1)
}
#XGboosr model

xgb_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

#Randomized search
random_search = RandomizedSearchCV(xgb_model, param_distributions=parameters, 
                                   n_iter=1000, scoring='neg_mean_squared_error', 
                                   cv=5, verbose=1, n_jobs=-1, random_state=42)
random_search.fit(X_train, y_train)

# Print best parameters
best_params = random_search.best_params_
print("Best hyperparameters:", best_params)

# Train model with best parameters
model = xgb.XGBRegressor(**best_params, objective='reg:squarederror', random_state=42)
model.fit(X_train, y_train)

# Feature importance plot
xgb.plot_importance(model)
plt.show()
#Evaluate performance trained model with test data
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

#print results
print("MSE:", mse)
print("MAE:", mae)
print("Hyper parameters:", best_params)
print("Feature importance:", model.feature_importances_)