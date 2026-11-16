#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install scikit-learn')


# # Apply for Multivariate Model without slicing

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Bandwidth_synthetic_data_1.csv")

# Select relevant features and target variable
features = ['RSSI', 'RSRQ', 'Speed', 'TA', 'LTE_Neighbour', 'Band', 'Echng']
target = 'Bandwidth'

# Drop rows with missing values
df = df.dropna()

# Define Features (X) and Target (y)
X = df[features]
y = df[target]

# Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and Train Random Forest Model with improved hyperparameters
rf = RandomForestRegressor(
    n_estimators=300,        # Increase number of trees
    max_depth=30,            # Deeper trees for better learning
    min_samples_split=5,     # Reduce split threshold
    min_samples_leaf=2,      # Prevent overfitting
    max_features=0.8,        # Use 80% of features per split
    bootstrap=True,          # Bootstrap sampling
    random_state=42
)

# Train the model
rf.fit(X_train, y_train)

# Predictions
y_pred = rf.predict(X_test)

# Calculate Performance Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Compute Pearson Correlation (CORR)
corr = np.corrcoef(y_test, y_pred)[0, 1]

# Print Results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Correlation Coefficient (R²): {r2:.4f}")
print(f"Pearson Correlation (CORR): {corr:.4f}")


# # Apply for Unitivariate Model without slicing

# In[2]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Bandwidth_synthetic_data_1.csv")

# List of individual features and target
features = ['RSSI', 'RSRQ', 'Speed', 'TA', 'LTE_Neighbour', 'Band', 'Echng']
target = 'Bandwidth'

# Drop missing values
df = df.dropna()

# Loop through each feature for univariate prediction
for feature in features:
    print(f"\n🔍 Evaluating univariate model using only: {feature}")
    
    # Define feature and target
    X = df[[feature]]  # Use double brackets to keep X as DataFrame
    y = df[target]

    # Split into train/test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=1.0,  # Since univariate, use full feature
        bootstrap=True,
        random_state=42
    )
    rf.fit(X_train, y_train)

    # Predict
    y_pred = rf.predict(X_test)

    # Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    # Print results
    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Pearson CORR: {corr:.4f}")



# # Apply for Unitivariate Model with slicing

# In[3]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Bandwidth_synthetic_data_1.csv")

# Features and target
features = ['RSSI', 'RSRQ', 'Speed', 'TA', 'LTE_Neighbour', 'Band', 'Echng']
target = 'Bandwidth'

# Drop missing values
df = df.dropna()

# Define slicing function (by RSRQ level)
def slice_rsrq(row):
    if row['RSRQ'] <= -14:
        return 'Low'
    elif -14 < row['RSRQ'] <= -9:
        return 'Medium'
    else:
        return 'High'

df['RSRQ_Slice'] = df.apply(slice_rsrq, axis=1)

# Loop through each slice
for slice_label in ['Low', 'Medium', 'High']:
    print(f"\n📶 Training models for slice: {slice_label}")

    slice_df = df[df['RSRQ_Slice'] == slice_label]

    if len(slice_df) < 50:
        print("  Skipping slice due to insufficient data.")
        continue

    for feature in features:
        print(f"\n🔍 Univariate model using: {feature}")
        
        # Prepare data
        X = slice_df[[feature]]
        y = slice_df[target]

        # Train/test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Model
        rf = RandomForestRegressor(
            n_estimators=300,
            max_depth=30,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features=1.0,
            bootstrap=True,
            random_state=42
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        corr = np.corrcoef(y_test, y_pred)[0, 1]

        # Output
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Pearson CORR: {corr:.4f}")
       


# # Apply for Multivariate Model with slicing

# In[4]:


import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Bandwidth_synthetic_data_1.csv")

# Drop missing values
df = df.dropna()

# Simulate Network Slicing by Speed
def classify_slice(speed):
    if speed <= 30:
        return 'Low'
    elif speed <= 60:
        return 'Medium'
    else:
        return 'High'

df['Slice'] = df['Speed'].apply(classify_slice)

# Define features and target
features = ['RSSI', 'RSRQ', 'Speed', 'TA', 'LTE_Neighbour', 'Band', 'Echng']
target = 'Bandwidth'

# Store results
results = {}

for slice_name in df['Slice'].unique():
    print(f"\n📶 Training model for slice: {slice_name}")
    
    # Filter data for this slice
    df_slice = df[df['Slice'] == slice_name]
    
    X = df_slice[features]
    y = df_slice[target]
    
    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Random Forest Regressor
    rf = RandomForestRegressor(
        n_estimators=300,
        max_depth=30,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=0.8,
        bootstrap=True,
        random_state=42
    )
    
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Evaluation Metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    corr = np.corrcoef(y_test, y_pred)[0, 1]

    results[slice_name] = {
        "MSE": mse, "RMSE": rmse, "MAE": mae, "R2": r2, "CORR": corr
    }

    print(f"  MSE: {mse:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    print(f"  Pearson CORR: {corr:.4f}")

# Optionally: Compare performance across slices
print("\n📊 Summary of Sliced Models:")
for slice_name, metrics in results.items():
    print(f"{slice_name} → R²: {metrics['R2']:.4f}, CORR: {metrics['CORR']:.4f}")


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




