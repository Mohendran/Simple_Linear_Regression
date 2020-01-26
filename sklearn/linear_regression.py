import pandas as pd
import matplotlib.pyplot as plt

# Convert CSV to a dataset
# It is of type DataFrame
# TODO: Need to read about DataFrames
dataset = pd.read_csv('../Data/Salary_Data.csv')

# Convert DataFrame to arrays
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 1].values

# Split up training Data and testing Data
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

# TODO: Learn to use this. What is happening?
# X is 2 dimensional & Y is 1 dimensional
"""from sklearn.preprocessing import StandardScaler
scaler = StandardScaler();
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
#Y_train = scaler.fit_transform(Y_train)
#Y_test = scaler.fit_transform(Y_test)
plt.plot(X_train,Y_train)"""

# Convert the training Data to a Model(Regressor)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predict the Salary with Experience(Testing Data)
Y_pred = regressor.predict(X_test)

# Plot the Training Data
plt.scatter(X_train, Y_train, color="Yellow")

# Plot the Testing Data to check whether the Line overlaps with points
plt.scatter(X_test, Y_test, color = 'Green')

# Plot the line, generated from the model
plt.plot(X_train, regressor.predict(X_train), color="Blue")

# Plot the predictions for the Testing Salary
plt.scatter(X_test, Y_pred, color="Red")

# Make the Graph readable
plt.title('Salary vs Experience')
plt.xlabel('Experience')
plt.ylabel('Salary')
