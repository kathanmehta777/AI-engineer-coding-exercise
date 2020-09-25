'''
Choice of regression:

I have chosen MLP regression with PCA for this assignment. I chose this after 
testing different models like linear regression, random forest regression, 
polynomial regression and MLP regression. I divided the training data into train
and validation sets and used the validation set to figure out the best model.
The moel that gave the least rmse on validation data is MLP regression with PCA.
I used this model to do the predictions on test data as well. The best model is 
trained on 80% on training data.

I have used PCA to reduce the number of features from 28 to 7. This can be 
inferred from the scree plot.
'''

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.neural_network import MLPRegressor


# Scree plot to better understand PCA
def scree_plot(pca_model):
    # The following code constructs the Scree plot
    per_var = np.round(pca_model.explained_variance_ratio_ * 100, decimals=1)
    labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

    plt.bar(x=range(1, len(per_var)+1), height=per_var, tick_label=labels)
    plt.ylabel('Percentage of Explained Variance')
    plt.xlabel('Principal Component')
    plt.title('Scree Plot')
    plt.show()

# Center and scale the data


def scale_data(data):
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(data)
    return scaler

# Create a PCA object


def pca_fit(n_comp, data):
    pca = PCA(n_components=n_comp)
    # Fit on training set only.
    pca.fit(data)
    return pca

# Applying PCA on entire dataset


def apply_pca(train_data, test_data, n_comp=None):
    # Center and scale the data
    scaler = scale_data(train_data)
    # Apply transform to both the training set and the test set.
    scaled_train_data = scaler.transform(train_data)
    scaled_test_data = scaler.transform(test_data)

    # Create a PCA object
    pca = pca_fit(n_comp, scaled_test_data)
    # Apply transform to both the training set and the test set.
    pca_train_data = pca.transform(scaled_train_data)
    pca_test_data = pca.transform(scaled_test_data)

    # Draw a scree plot of principal components
    scree_plot(pca)

    return pca_train_data, pca_test_data, scaler, pca

# Dividing dataset into train and validation dataset depending on per_divide


def divide_data(data, labels, per_divide):
    train_data, validation_data, train_labels, validation_labels = train_test_split(data, labels,
                                                                                    test_size=per_divide)
    return train_data, train_labels, validation_data, validation_labels

# Evaluating models on 3 metrics - MAE, MSE, RMSE


def evaluate(actual_labels, pred_labels, verbose=True):
    mae = metrics.mean_absolute_error(actual_labels, pred_labels)
    mse = metrics.mean_squared_error(actual_labels, pred_labels)
    rmse = np.sqrt(mse)
    if (verbose):
        print('Mean Absolute Error:', mae)
        print('Mean Squared Error:', mse)
        print('Root Mean Squared Error:', rmse)
    return rmse

# Linear Regression model


def my_model1(train_data, train_labels, val_data):
    print("===========================LINEAR REGRESSION===========================")
    regressor = LinearRegression()
    model = regressor.fit(train_data, train_labels)
    pred_train = regressor.predict(train_data)
    print("''''''''Train errors''''''''")
    evaluate(train_labels, pred_train)
    pred_val = regressor.predict(val_data)
    return pred_val, regressor

# Polynomial Regression model


def my_model2(train_data, train_labels, val_data):
    print("===========================POLYNOMIAL REGRESSION===========================")
    input = [('polynomial', PolynomialFeatures(degree=2)),
             ('modal', LinearRegression())]
    regressor = Pipeline(input)
    regressor.fit(train_data, train_labels)
    pred_train = regressor.predict(train_data)
    print("''''''''Train errors''''''''")
    evaluate(train_labels, pred_train)
    pred_test = regressor.predict(val_data)
    return pred_test, regressor

# Random forest Regressor


def my_model3(train_data, train_labels, test_data):
    print("===========================RANDOM FOREST REGRESSION===========================")
    regressor = RandomForestRegressor(n_estimators=10)
    regressor.fit(train_data, train_labels)
    pred_train = regressor.predict(train_data)
    print("''''''''Train errors''''''''")
    evaluate(train_labels, pred_train)
    pred_test = regressor.predict(test_data)
    return pred_test, regressor

# Neural Network Regressor


def my_model4(train_data, train_labels, val_data):
    print("===========================NEURAL NETWORK REGRESSION===========================")

    regressor = MLPRegressor(
        hidden_layer_sizes=(16, 64, 128, 128, 64, 32),  activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
        learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=200, shuffle=True,
        random_state=0, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
        early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    regressor.fit(train_data, train_labels)
    pred_train = regressor.predict(train_data)
    print("''''''''Train errors''''''''")
    evaluate(train_labels, pred_train)
    pred_test = regressor.predict(val_data)
    return pred_test, regressor

# Running the best model out of the 4 models


def best_model(data, model, pca, scaler):
    scaled_data = scaler.transform(data)
    transformed_data = pca.transform(scaled_data)
    y_pred = model.predict(transformed_data)
    return y_pred


if __name__ == "__main__":
    # Load data
    training_data = np.load("./data/ml/train_swir_nr.npy")
    labels = np.load("./data/ml/train_concentration.npy")
    test_data = np.load("./data/ml/test_swir_nr.npy")

    train_data, train_labels, validation_data, validation_labels = divide_data(
        training_data, labels, 0.2)
    test_errors = []
    names = ['my_model1', 'my_model2', 'my_model3', 'my_model4']

    # Linear Regression
    pred_test1, model1 = my_model1(train_data, train_labels, validation_data)
    print("''''''''Test errors''''''''")
    error_model1 = evaluate(validation_labels, pred_test1)
    test_errors.append(error_model1)
    print()

    # Polynomial regression
    pred_test2, model2 = my_model2(train_data, train_labels, validation_data)
    print("''''''''Test errors''''''''")
    error_model2 = evaluate(validation_labels, pred_test2)
    test_errors.append(error_model2)
    print()

    # Random Forest Regression
    pred_test3, model3 = my_model3(train_data, train_labels, validation_data)
    print("''''''''Test errors''''''''")
    error_model3 = evaluate(validation_labels, pred_test3)
    test_errors.append(error_model3)
    print()

    # Neural Network regression
    pred_test4, model4 = my_model4(
        train_data, train_labels, validation_data)
    print("''''''''Test errors''''''''")
    error_model4 = evaluate(validation_labels, pred_test4)
    test_errors.append(error_model4)
    print()

    # Printing 25 samples of predicted actual_shopping_duration_min of each model and comparing it with ground truth value
    df = pd.DataFrame({'Actual': validation_labels, 'Predicted_1': pred_test1,
                       'Predicted_2': pred_test2, 'Predicted_3': pred_test3,
                       'Predicted_4': pred_test4})
    print(df.head(25))

    # Plotting RMSE of each model obtained on the validation dataset
    plt.plot(names, test_errors, color='green', linestyle='dashed', linewidth=3,
             marker='o', markerfacecolor='blue', markersize=12)
    plt.show()

    ''' NOTE: Seeing the above analysis and the plot on validation error, 
    it can be concluded that model 4(Neural Network regression) is the best 
    model'''
    # Applying PCA on the best model
    comp = 7
    pca_train_data, pca_val_data, scaler, pca = apply_pca(
        train_data, validation_data, comp)
    print("===========BEST MODEL=============")
    pred_test4, model4 = my_model4(
        pca_train_data, train_labels, pca_val_data)
    print("''''''''Test errors''''''''")
    error_model4 = evaluate(validation_labels, pred_test4)

    # Predicting on training data
    y_pred = best_model(training_data, model4, pca, scaler)

    # Plotting y_pred vs y
    fig, ax = plt.subplots()
    ax.scatter(y_pred, labels)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Measured')
    plt.show()

    # Predicting on test data
    y_pred_test = best_model(test_data.T, model4, pca, scaler)
    np.save("./data/ml/y_pred_test.npy", y_pred_test)
