import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error

data_path = "../data/"

df = pd.read_fwf(data_path + "brain_body.txt")
print(df.info())
print(df.head())
print(df.shape)

X = df[['Brain']]
y = df[['Body']]

X_train, X_test, y_train, y_test = train_test_split(X,y)

model = linear_model.LinearRegression()
model.fit(X_train,y_train)

print(mean_squared_error(y_test, model.predict(X_test)))

# plt.scatter(X, y)
# plt.plot(X, model.predict(X))
# plt.show()


