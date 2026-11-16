#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system('pip install tensorflow')


# In[2]:


get_ipython().system('pip install scikit-learn')


# In[3]:


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("Bandwidth_synthetic_data_1.csv")

# Select relevant features and target variable
features = ['RSSI', 'RSRQ', 'Speed', 'TA', 'LTE_Neighbour', 'Band', 'Echng']
target = 'Bandwidth'

# Drop missing values
df = df.dropna()

# Define Features (X) and Target (y)
X = df[features].values
y = df[target].values

# Scale Features (LSTMs work better with normalized data)
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X = scaler_X.fit_transform(X)
y = scaler_y.fit_transform(y.reshape(-1, 1))  # Reshape y for scaling

# Split data into Training (80%) and Testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Reshape X for LSTM input (samples, timesteps=1, features)
X_train = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
X_test = X_test.reshape((X_test.shape[0], 1, X_test.shape[1]))

# Build Vanilla LSTM Model
model = Sequential([
    LSTM(50, activation='relu', input_shape=(1, X_train.shape[2])),
    Dense(1)  # Output layer (single value prediction)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=20, batch_size=64, verbose=1)

# Predictions
y_pred = model.predict(X_test)

# Inverse transform the scaled values to original scale
y_test = scaler_y.inverse_transform(y_test)
y_pred = scaler_y.inverse_transform(y_pred)

# Compute Performance Metrics
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Compute Pearson Correlation
corr = np.corrcoef(y_test.flatten(), y_pred.flatten())[0, 1]

# Print Results
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"Correlation Coefficient (R²): {r2:.4f}")
print(f"Pearson Correlation (CORR): {corr:.4f}")


# In[ ]:





# In[ ]:





# In[6]:





# In[7]:





# In[8]:





# In[ ]:





# In[ ]:




