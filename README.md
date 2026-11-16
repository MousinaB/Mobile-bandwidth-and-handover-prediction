# Mobile Bandwidth Prediction and Handover Prediction

## Project Overview

This project focuses on improving mobile network performance through Machine Learning and Deep Learning techniques. It consists of two major components: **Mobile Bandwidth Prediction** and **Network Handover Prediction**.

The objective is to predict network bandwidth based on radio and network parameters and identify potential handover events using machine learning classification models.

The project uses synthetic mobile network datasets to experiment with different predictive models and evaluate their performance.

## Project Objectives

- Predict mobile network bandwidth using network-related parameters.
- Analyze the relationship between signal quality, user mobility and bandwidth.
- Predict network handover events using radio and network measurements.
- Compare traditional Machine Learning and Deep Learning approaches.
- Explore the impact of bandwidth-related features on handover prediction.

## Project Structure

```text
Project-1/
│
├── Part-A/
│   ├── Bandwidth_synthetic_data (1).py
│   ├── Bandwidth_synthetic_data (1).ipynb
│   ├── RLS Method.py
│   ├── RLS Method.ipynb
│   ├── Random forest.py
│   ├── Random forest.ipynb
│   ├── Vanilla_LSTM.py
│   ├── Vanilla_LSTM.ipynb
│   ├── TPA_LSTM.py
│   └── TPA_LSTM.ipynb
│
├── Part-B/
│   ├── Handover_synthetic_data.py
│   ├── Handover_synthetic_data.ipynb
│   ├── random_forest_handover.py
│   └── gb-method-handover.ipynb
│
└── README.md
```

## Part A: Mobile Bandwidth Prediction

This component predicts mobile network bandwidth using network measurements.

### Input Features

- RSSI – Received Signal Strength Indicator
- RSRQ – Reference Signal Received Quality
- Speed – User mobility speed
- TA – Timing Advance
- LTE_Neighbour – Neighbouring LTE cell information
- Band – Network frequency band
- Echng – Network-related parameter

### Target Variable

- Bandwidth

### Models Implemented

1. **Recursive Least Squares (RLS):** An adaptive estimation technique for bandwidth prediction.
2. **Random Forest Regression:** Predicts bandwidth using multiple network parameters.
3. **Vanilla LSTM:** Uses sequential learning to model temporal patterns in network bandwidth.
4. **TPA-LSTM:** Combines LSTM with temporal pattern attention mechanisms for sequence-based prediction.

### Experiments

- Univariate and multivariate bandwidth prediction.
- Network slicing-based prediction.
- Feature-wise model evaluation.
- Performance comparison using different prediction approaches.

## Part B: Mobile Network Handover Prediction

This component focuses on predicting mobile network handover events based on network conditions and radio measurements.

### Input Features

- DL – Downlink bandwidth
- UL – Uplink bandwidth
- RSSI – Received Signal Strength Indicator
- RSRQ – Reference Signal Received Quality
- RSRP – Reference Signal Received Power
- NRXSRP – Neighbouring cell RSRP
- NRXSRQ – Neighbouring cell RSRQ
- SNR – Signal-to-Noise Ratio
- CQI – Channel Quality Indicator
- Speed – User mobility speed

### Target Variable

- Handover – Handover event classification.

### Models Implemented

1. **Random Forest Classifier:** Classifies handover events using network and radio parameters.
2. **XGBoost Classifier:** Uses gradient boosting for handover prediction.

### Experiments

- Synthetic handover data generation.
- Handling class imbalance using SMOTE.
- Model training and testing.
- Handover prediction using different feature combinations.
- Comparing prediction performance with and without bandwidth-related features.
- ROC curve and correlation analysis.

## Technologies and Libraries

- **Programming Language:** Python
- **Machine Learning:** Scikit-learn
- **Deep Learning:** TensorFlow, Keras
- **Models:** Random Forest, XGBoost, LSTM, TPA-LSTM, RLS
- **Data Processing:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Data Balancing:** SMOTE
- **Development Environment:** Jupyter Notebook, Google Colab

## Evaluation Metrics

The project uses different evaluation metrics depending on the prediction task.

**Bandwidth Prediction (Regression):**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Pearson Correlation

**Handover Prediction (Classification):**
- Accuracy
- Classification Report
- ROC Curve
- Area Under the ROC Curve (AUC)

## Installation and Setup

Clone the repository:

```bash
git clone <your-github-repository-url>
cd Project-1
```

Install the required dependencies:

```bash
pip install numpy pandas matplotlib seaborn
pip install scikit-learn tensorflow xgboost imbalanced-learn scipy
```

Open the notebooks using Jupyter Notebook or Google Colab.

## Dataset

This project uses synthetically generated mobile network bandwidth and handover datasets.

Run the corresponding synthetic data generation notebooks or Python scripts before executing the prediction models.

Note: The generated CSV datasets are not included in the current repository ZIP.

## Key Highlights

- Implementation of multiple Machine Learning and Deep Learning algorithms.
- Regression-based mobile bandwidth prediction.
- Classification-based network handover prediction.
- Sequential time-series modelling using LSTM architectures.
- Network slicing-based bandwidth prediction experiments.
- Class imbalance handling using SMOTE.
- Feature importance and comparative model analysis.

## Conclusion

This project explores Machine Learning and Deep Learning approaches for mobile network performance prediction. By combining bandwidth estimation and handover classification, it investigates how network measurements and user mobility can support intelligent network management and improved connectivity.


