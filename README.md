# 🏠 Perceptron for House Price Prediction

This project focuses on applying machine learning techniques—including a custom-built **Perceptron**—to analyze and predict house prices using the **California Housing Dataset**. It includes data preprocessing, exploratory visualization, and the application of several models: a custom Perceptron, linear regression, and a deep neural network built with Keras.

---

## 📁 Dataset

The dataset (`housing.csv`) includes information on California housing blocks such as:
- Median income
- House age
- Total rooms
- Population
- Median house value (target)
- And more, including a categorical feature (`ocean_proximity`)

---

## 📊 Features of the Project

### 1. Data Exploration and Visualization
- Dataset structure, null value handling, and statistical summaries
- Histograms and KDE plots for feature distribution
- Scatter plots showing feature relationships and geographic clustering

### 2. Preprocessing Steps
- Handling missing values (`total_bedrooms` filled with median)
- Feature scaling using:
  - MinMaxScaler
  - StandardScaler
  - MaxAbsScaler
  - RobustScaler
  - Normalizer
- One-Hot Encoding for categorical variables

### 3. Machine Learning Models

#### 🧠 Custom Perceptron
- Implemented from scratch
- Trained using k-fold cross-validation
- Evaluated using:
  - Classification Accuracy
  - Mean Squared Error (MSE)
  - Mean Absolute Error (MAE)

#### 📈 Linear Regression (from scratch)
- Univariate regression using 10-fold CV
- Manually calculated slope/intercept
- Metrics:
  - MSE
  - MAE
- Final predictions plotted for visual inspection

#### 🤖 Deep Neural Network (Keras)
- Multilayer feedforward neural network
- Layers: Dense(128) → Dense(32) → Dense(8) → Dense(1)
- Loss: Mean Squared Error
- Optimizer: Adam
- Early stopping to avoid overfitting
- R² score printed for validation performance

---

## 📦 Requirements

Install the necessary packages using pip:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn keras tensorflow
