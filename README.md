# Crop Recommendation Model

## Overview
This repository contains a machine learning solution that helps farmers select optimal crops based on various agricultural parameters. By analyzing soil nutrients, climate conditions, and other environmental factors, the model recommends crops most likely to thrive in specific conditions, ultimately maximizing yield while minimizing inputs and resource wastage.

## Problem Statement
Farmers often invest significant capital and resources without knowledge of which crops would optimally thrive in their specific conditions. This leads to:
- Suboptimal yields
- Wasted resources (fertilizers, water, etc.)
- Financial losses
- Environmental impact from excessive fertilizer use

This model aims to solve these problems by providing data-driven crop recommendations based on scientific analysis of environmental and soil conditions.

## Features
- **Soil Analysis**: Utilizes NPK (Nitrogen, Phosphorus, Potassium) values to match crops with appropriate soil nutrient profiles
- **Climate Matching**: Factors in temperature and humidity data to ensure crops are suitable for local climate conditions
- **Water Management**: Incorporates rainfall data to recommend crops with appropriate water requirements
- **Soil pH Optimization**: Matches crops to appropriate soil pH levels
- **Multi-Factor Analysis**: Combines all parameters to provide holistic recommendations

## Dataset
The model is trained on a comprehensive dataset that includes:
- Soil nutrient levels (N, P, K)
- Temperature data
- Humidity measurements
- pH values
- Rainfall statistics
- Crop labels (target variable)

## Technical Approach

### Data Preprocessing
```python
# Checking Missing Values
df.isnull().sum()

# Handling Categorical Values Using One-Hot Encoder
from sklearn.preprocessing import OneHotEncoder

encoder = OneHotEncoder(sparse_output=False, drop='first')
encoded_data = encoder.fit_transform(df[categorical_cols])

encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(categorical_cols)
)

final_df = pd.concat([df.drop(categorical_cols, axis=1), encoded_df], axis=1)
```

### Feature Engineering
```python
# Nutrient ratios
df['N_P_ratio'] = df['N'] / df['P']
df['N_K_ratio'] = df['N'] / df['K']
df['P_K_ratio'] = df['P'] / df['K']

# NPK composite score 
df['npk_score'] = (df['N'] * 0.4) + (df['P'] * 0.3) + (df['K'] * 0.3)

# pH suitability
optimal_ph = {
    'wheat': (6.0, 7.0),
    'rice': (5.0, 6.5),
    'corn': (5.8, 6.8),
}

def ph_suitability(row):
    crop = row['label']
    min_ph, max_ph = optimal_ph.get(crop, (0, 14))
    return 1 if min_ph <= row['ph'] <= max_ph else 0

df['ph_suitability'] = df.apply(ph_suitability, axis=1)
```

### Model Pipeline
```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Building preprocessing pipeline
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ('num', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ]), numeric_features),
        ('cat', Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ]), categorical_features)
    ])

# Full pipeline with model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier())
])
```

## Installation

### Requirements
- Python 3.8+
- pandas
- numpy
- scikit-learn
- matplotlib
- seaborn
- lime (for model interpretability)

### Setup
1. Clone this repository:
   ```
   git clone https://github.com/Sigilai-hacks/Capstone_Project.git
   cd Capstone_Project.ipynb
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the setup script:
   ```
   python setup.py
   ```

## Usage

### Basic Usage
```python
from crop_recommender import CropRecommender

# Initialize the model
recommender = CropRecommender()

# Load the trained model
recommender.load_model('models/random_forest_model.pkl')

# Get a recommendation
prediction = recommender.predict({
    'N': 90,
    'P': 42,
    'K': 43,
    'temperature': 20.87,
    'humidity': 82.00,
    'ph': 6.5,
    'rainfall': 202.93
})

print(f"Recommended crop: {prediction}")
```

## Model Performance

Our evaluation shows that Random Forest consistently outperformed other models:

| Model | Accuracy | F1-Score | ROC-AUC |
|-------|----------|----------|---------|
| Random Forest | 0.97 | 0.96 | 0.99 |
| Decision Tree | 0.93 | 0.92 | 0.97 |
| Logistic Regression | 0.85 | 0.84 | 0.92 |
| SVM | 0.91 | 0.90 | 0.95 |

## Feature Importance

The most important features for crop prediction were:
1. Rainfall
2. Temperature
3. Nitrogen (N) content
4. pH value
5. Potassium (K) content

## Project Structure

```
├── data/
│   ├── raw/                  # Raw data files
│   ├── processed/            # Processed data files
│   └── external/             # External reference data
├── models/                   # Stored trained models
├── notebooks/                # Jupyter notebooks for exploration
├── src/                      # Source code
│   ├── data/                 # Data processing scripts
│   ├── features/             # Feature engineering scripts
│   ├── models/               # Model training scripts
│   └── visualization/        # Visualization scripts
├── app/                      # Web application
├── tests/                    # Unit tests
├── requirements.txt          # Dependencies
├── setup.py                  # Setup script
└── README.md                 # This file
```

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Future Improvements

1. **Real-time Weather Integration**: Connect with weather APIs for real-time data
2. **Soil Sensor Integration**: Develop interfaces for direct soil sensor data input
3. **Mobile Application**: Create a mobile app for field use
4. **Expanded Crop Database**: Include more crops and regional varieties
5. **Economic Analysis**: Add cost-benefit analysis for recommended crops

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- Data was sourced from KAGGLE
- Inspired by the need for sustainable and efficient agricultural practices
