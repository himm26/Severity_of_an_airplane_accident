# Severity of an Airplane Accident
Building Machine Learning models to anticipate and classify the severity of any airplane accident based on past incidents.

# Dataset
Dataset containing the information of past incidents of airplanes having the following parameters: 
- Weather,
- Cabin Temperature,
- Control Metric,
- Turbulence in Gforces,
- Total Violations,
- Days Since Inspection,
- Max Elevation etc.

Dataset Size: 1000x12

# Data Modeling
- Removing Null values and outliers
- Appling Z-score normalization  [z = (x – μ) /σ]

# Classification Models
- Decision Tree
- Random Forest 
- SVM

Used RandomizedSearchCV for the tuning of Hyperparameters.

# Accuracy 
- 94.5% with Random Forest, 
- 88.8% with SVM, 
- 87.4 with Decision Tree.
