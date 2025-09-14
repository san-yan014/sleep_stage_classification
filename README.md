# Sleep Stage Classification Using Wearable Device Data

## Project Overview
This project implements machine learning models to classify sleep stages (Wake, N1, N2, N3, REM) using physiological data from wearable devices. The work compares traditional machine learning approaches with deep learning methods for non-invasive sleep monitoring.

## Dataset
- **Source**: DREAMT dataset from PhysioNet (100+ participants)
- **Data Type**: Physiological time series from Empatica E4 wearables
- **Sensors**: 8 channels (BVP, ACC_X/Y/Z, TEMP, EDA, HR, IBI)
- **Sampling Rate**: 64 Hz
- **Window Size**: 30-second epochs (1920 samples per window)
- **Labels**: Expert-annotated sleep stages from polysomnography

## Project Structure
```
sleep-classification/
├── data_processing/
│   ├── data_loader.py          # Load and preprocess DREAMT data
│   ├── windowing.py            # Create 30-second windows
│   └── feature_extraction.py   # Extract statistical features
├── models/
│   ├── xgboost_baseline.py     # Traditional ML baseline
│   └── cnn_model.py            # 1D CNN implementation
├── evaluation/
│   └── metrics.py              # Performance evaluation
├── notebooks/
│   └── analysis.ipynb          # Data exploration and results
└── README.md
```

## Methodology

### Data Preprocessing
1. **Missing Value Handling**: IBI missing values filled with 0 followed by interpolation
2. **Data Cleaning**: Removed windows with "Missing" sleep stage labels
3. **Label Processing**: Converted "P" (preparation) stages to "W" (wake)
4. **Windowing**: Grouped sensor data into 30-second classification windows
5. **Normalization**: Applied StandardScaler for feature scaling

### Models Implemented

#### 1. XGBoost Baseline
- **Features**: 64 statistical features (8 per sensor)
  - Mean, max, min, 25th/75th percentiles, range, variance, standard deviation
- **Architecture**: Gradient boosted decision trees
- **Performance**: 80% accuracy on test set

#### 2. 1D Convolutional Neural Network
- **Input**: Raw sensor sequences (8 channels × 1920 timesteps)
- **Architecture**:
  - Conv1D layers with ReLU activation and MaxPooling
  - Dropout layers for regularization
  - Fully connected layers for classification
- **Framework**: PyTorch

## Key Findings

### XGBoost Results
- **Overall Accuracy**: 80%
- **Best Performing Class**: N3 (Deep Sleep) - 85% precision despite limited training data
- **Challenging Class**: N1 (Light Sleep) - 8% recall due to transitional nature
- **Insight**: Deep sleep exhibits more distinctive physiological patterns, making it easier to classify despite fewer samples

### CNN Development
- Implemented 1D CNN architecture for temporal pattern recognition
- Explored various hyperparameter configurations
- Currently achieving 67% validation accuracy
- Investigating data scaling and architecture optimization

## Technical Implementation

### Dependencies
```python
numpy
pandas
scikit-learn
xgboost
torch
scipy
matplotlib
seaborn
```

### Data Pipeline
```python
# Load and process participant data
X_raw, y = process_participants_to_arrays(data_path)

# Feature extraction for XGBoost
X_features = create_feature_dataset(X_raw)

# Prepare data for CNN
X_pytorch = np.transpose(X_raw, (0, 2, 1))  # Reshape for PyTorch
```

### Model Training
```python
# XGBoost
model = xgb.XGBClassifier(objective='multi:softmax', eval_metric='mlogloss')
model.fit(X_train_scaled, y_train)

# CNN
model = SleepCNN()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)
```

## Results Summary

| Model | Accuracy | Precision | Recall | F1-Score |
|-------|----------|-----------|---------|----------|
| XGBoost | 80% | 0.78 | 0.80 | 0.79 |
| 1D CNN | 67% | - | - | - |

## Clinical Insights
The classification results revealed important sleep physiology patterns:
- N3 (deep sleep) had the most distinctive and consistent patterns across participants
- N1 performed poorly due to its transitional nature between wake and sleep states
- The model successfully captured physiological differences between sleep stages using only wearable sensor data

## Future Work
- Optimize CNN architecture and hyperparameters
- Implement hybrid methods combining XGBoost and deep learning models
- Investigate attention mechanisms for identifying critical time periods
- Extend to sleep disorder detection applications

## Usage

### Training XGBoost Model
```bash
python models/xgboost_baseline.py --data_path /path/to/dreamt/data
```

### Training CNN Model
```bash
python models/cnn_model.py --epochs 20 --batch_size 64 --lr 0.0005
```

## References
- Wang, K., et al. (2025). DREAMT: Dataset for Real-time sleep stage EstimAtion using Multisensor wearable Technology. PhysioNet.
- Sleep stage classification literature and methodologies

## License
This project is for educational and research purposes. Please cite the DREAMT dataset appropriately when using this code.

## Contact
For questions about this implementation, please refer to the code documentation or create an issue in this repository.
