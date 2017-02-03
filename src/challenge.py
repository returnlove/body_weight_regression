import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
import numpy as np

data_path = "../data/"

df = pd.read_csv(data_path + "challenge_dataset.txt", names = ['X', 'y'])
print(df.info())
print(df.head())
print(df.shape)

X = df[['X']]
y = df[['y']]

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

print('MSE built in')
print(mean_squared_error(y_test, model.predict(X_test)))

print('MSE manual')
print(np.mean((model.predict(X_test) - y_test) ** 2))

# plt.scatter(X, y)
# plt.plot(X, model.predict(X))
# plt.show()


