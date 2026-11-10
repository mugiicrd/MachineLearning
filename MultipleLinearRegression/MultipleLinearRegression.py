# importing libraries for this task!
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# loading data into this variable "df"
df = pd.read_csv("MLR with Outliers and Missing Values By Siddant007.csv")

model = LinearRegression()

print(df.isnull().sum())

df = df.dropna()

X = df[["HouseSize", "DistanceToCity"]]

y = df["Price"]


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)

model.fit(X_train, y_train)

print("Intercept: ", model.intercept_)
print("Ceofficients: ", model.coef_)

y_pred = model.predict(X_test)

print("Predictions:", y_pred)

print("MAR: ", mean_absolute_error(y_test, y_pred))
print("MSR: ", mean_squared_error(y_test, y_pred))
print("R2: ", r2_score(y_test, y_pred))

plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("HouseDistance")
plt.ylabel("Salary")
plt.show()







