import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
import seaborn as sns
from scipy import stats
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


cal_housing = pd.read_csv('housing.csv')

# 1 Explore Data

print(cal_housing.info())
print(cal_housing.head())
print(cal_housing.describe())
print(cal_housing["ocean_proximity"].value_counts())


# 2 Scaling

# for the normalizer
median_value = cal_housing['total_bedrooms'].median()
cal_housing['total_bedrooms'].fillna(median_value, inplace=True)
number_columns = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
cal_housing_numbers = cal_housing[number_columns]


min_max_test = preprocessing.MinMaxScaler()
X_train_minmax_test = min_max_test.fit_transform(cal_housing_numbers)
cal_housing_minmax_scaled_df = pd.DataFrame(X_train_minmax_test, columns=number_columns)
print(cal_housing_minmax_scaled_df.head())

standar_test = preprocessing.StandardScaler()
X_train_standar_test = standar_test.fit_transform(cal_housing_numbers)
cal_housing_standard_scaled_df = pd.DataFrame(X_train_standar_test, columns=number_columns)
print(cal_housing_standard_scaled_df.head())

maxabs_test = preprocessing.MaxAbsScaler()
X_train_maxabs_test = maxabs_test.fit_transform(cal_housing_numbers)
cal_housing_maxabs_scaled_df = pd.DataFrame(X_train_maxabs_test, columns=number_columns)
print(cal_housing_maxabs_scaled_df.head())


robust_test = preprocessing.RobustScaler()
X_train_robust_test = robust_test.fit_transform(cal_housing_numbers)
cal_housing_robust_scaled_df = pd.DataFrame(X_train_robust_test, columns=number_columns)
print(cal_housing_robust_scaled_df.head())


normal_test = preprocessing.Normalizer()
X_train_normal_test = normal_test.fit_transform(cal_housing_numbers)
cal_housing_minmax_scaled_df = pd.DataFrame(X_train_normal_test, columns=number_columns)
print(cal_housing_minmax_scaled_df.head())



# 3 One hot Vector

unique_ocean_proximity = cal_housing['ocean_proximity'].unique()
print(unique_ocean_proximity)

ohv_categories = cal_housing[["ocean_proximity"]]
encoder = OneHotEncoder()
housing_cat_ohv = encoder.fit_transform(ohv_categories)
array = housing_cat_ohv.toarray()
encoderDF = pd.DataFrame(array, columns=encoder.categories_)
dataset = pd.concat([cal_housing, encoderDF], axis=1)
dataset = dataset.drop(['ocean_proximity'], axis=1)
print(dataset)



# 4 median value

print(cal_housing.info())
print("Κενές τιμές πριν: ")
print(cal_housing['total_bedrooms'].isna().sum())
median_value = cal_housing['total_bedrooms'].median()
cal_housing['total_bedrooms'].fillna(median_value, inplace=True)
print("Κενές τιμές μετά: ")
print(cal_housing['total_bedrooms'].isna().sum())






# visualize 


cal_housing.hist(bins=50, figsize=(15,8.1))
plt.show()


features = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value']
for feature in features:
    plt.hist(cal_housing[feature], bins=50, density=True, alpha=0.6, color='b')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.show()



for feature in features:
    sns.displot(cal_housing[feature], kde=True, color='b')
    plt.title(f'Histogram of {feature}')
    plt.xlabel(feature)
    plt.ylabel('Probability density')
    plt.show()





# visual 2

cal_housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
    s=cal_housing["population"]/100, label="population", figsize=(10,7),
    c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
    sharex=False)
plt.title('California housing prices')
plt.legend()
plt.show()


plt.scatter(cal_housing["median_income"], cal_housing["median_house_value"], alpha=0.1)
plt.xlabel("Median Income")
plt.ylabel("Median House Value")
plt.title("Median Income per Median House Value")
plt.axis([0, 16, 0, 550000])
plt.show()


plt.scatter(cal_housing["total_rooms"], cal_housing["median_house_value"], alpha=0.1)
plt.xlabel("Total Rooms")
plt.ylabel("Median House Value")
plt.title("Total Rooms per Median House Value")
plt.show()






# Perceptron

from perceptron import Perceptron

if __name__ == "__main__":

    cal_housing = pd.read_csv('housing.csv')

    def accuracy(y_true, y_pred):
        accuracy = np.sum(y_true == y_pred) / len(y_true)
        return accuracy

    def mean_squared_error(y_true, y_pred):
        mse = np.mean((y_true - y_pred) ** 2)
        return mse

    def mean_absolute_error(y_true, y_pred):
        mae = np.mean(np.abs(y_true - y_pred))
        return mae


    X = cal_housing.iloc[:, :-2].values
    y = cal_housing.iloc[:, -2].values

    k = 5
    fold_size = len(X) // k

    accuracies = []
    mses = []
    maes = []

    # k-fold cross validation
    for i in range(k):
        test_start = i * fold_size
        test_end = (i + 1) * fold_size
        train_indices = list(range(test_start)) + list(range(test_end, len(X)))

        X_train, X_test = X[train_indices], X[test_start:test_end]
        y_train, y_test = y[train_indices], y[test_start:test_end]

        p = Perceptron(learning_rate=0.01, n_iters=1000)
        p.fit(X_train, y_train)
        predictions = p.predict(X_test)

        acc = accuracy(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)

        accuracies.append(acc)
        mses.append(mse)
        maes.append(mae)


    mean_accuracy = np.mean(accuracies)
    mean_mse = np.mean(mses)
    mean_mae = np.mean(maes)


    print("Mean classification accuracy:", mean_accuracy)
    print("Mean Squared Error:", mean_mse)
    print("Mean Absolute Error:", mean_mae)






# least squares
    
def kfold_indices(N, K):
    indices = np.arange(N)
    M = N // K
    if N % K != 0:
        raise ValueError("The number of elements within vector Indices must be fully divided by K")
    else:
        train_indices = []
        test_indices = []
        for k in range(K):
            start = k * M
            end = (k + 1) * M if k < K - 1 else N
            test_indices.append(indices[start:end])
            train_indices.append(np.setdiff1d(indices, test_indices[-1]))
    return train_indices, test_indices


def linear_regression_fit(X, Y, K=10):
    mse_scores = []
    mae_scores = []


    train_indices, test_indices = kfold_indices(len(X), K)

    for train_index, test_index in zip(train_indices, test_indices):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

        X_train_mean = np.mean(X_train)
        Y_train_mean = np.mean(Y_train)

        num = np.sum((X_train - X_train_mean) * (Y_train - Y_train_mean))
        den = np.sum((X_train - X_train_mean) ** 2)
        m = num / den
        c = Y_train_mean - (m * X_train_mean)

        Y_pred = m * X_test + c

        mse = np.mean((Y_test - Y_pred) ** 2)
        mae = np.mean(np.abs(Y_test - Y_pred))

        mse_scores.append(mse)
        mae_scores.append(mae)

    
    plt.scatter(X_test, Y_pred, color='red')
    return mse_scores, mae_scores


if __name__ == "__main__":

    X = cal_housing.iloc[:, 0]
    Y = cal_housing.iloc[:, 1]

    plt.scatter(X, Y)

    mse_scores, mae_scores = linear_regression_fit(X, Y)

    print("Mean Squared Error (MSE) Scores:", mse_scores)
    print("Mean Absolute Error (MAE) Scores:", mae_scores)
    print("Average MSE:", np.mean(mse_scores))
    print("Average MAE:", np.mean(mae_scores))

    
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Linear Regression Fit")
    plt.show()







# MultiLayer NN
    
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score


# Τα ξανακάνω απλά για να φανούν και εδώ
dataset = pd.read_csv('housing.csv')
median_value = dataset['total_bedrooms'].median()
dataset['total_bedrooms'].fillna(median_value, inplace=True)
dataset = pd.get_dummies(dataset, columns=['ocean_proximity'])




X = dataset.iloc[:,0:9]  
Y = dataset.iloc[:,9]

# Split the data into training and validation sets
X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.3, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Create model
model = Sequential([
    Dense(128, activation="relu", input_dim=X_train.shape[1]),
    Dense(32, activation="relu"),
    Dense(8, activation="relu"),
    Dense(1, activation="linear")
])

# Compile model
model.compile(loss='mean_squared_error', optimizer=Adam(learning_rate=1e-3, decay=1e-3 / 200))

# Patient early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)

# Fit the model
history = model.fit(X_train, Y_train, validation_data=(X_val, Y_val), epochs=10000000, batch_size=100, verbose=2, callbacks=[es])

# Calculate predictions
PredTestSet = model.predict(X_train)
PredValSet = model.predict(X_val)

# Plot loss history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.title('Trainig History')
plt.xlabel('Epoch')
plt.ylabel('Validation loss')
plt.legend()
plt.show()

# Compute R-Square value for validation set
ValR2Value = r2_score(Y_val, PredValSet)
print("Validation Set R-Square=", ValR2Value)

