#Importing libraries to make it all work
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

#connecting the data to variable df
df = pd.read_csv("Salary_dataset.csv")

#creating my model
model = LinearRegression()

#connecting the columns to variables X and y
X = df[["YearsExperience"]]
y = df["Salary"]

print(df.head())
print(df.info())
print(df.describe())

print("Missing values:")
print(df.isnull().sum())

print("Duplicates:", df.duplicated().sum())
# using the train test split function
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
# training the model using data
model.fit(X_train, y_train)

# making predictions based on 20% of data and getting R^2
print(model.predict(X_test))
print(model.score(X_test, y_test))

y_pred = model.predict(X_test)
# calculate MAE, MSE, R2
print("\nMODEL EVALUATION")
print("-------------------------")
print("R2:", r2_score(y_test, y_pred))
print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))

print("\nSALARY PREDICTION")
print("-----------------------")

years = float(input("Enter years of experience: "))

new_data = pd.DataFrame({"YearsExperience":[years]})

prediction = model.predict(new_data)


#creating pyplot so we can showcase the data
plt.scatter(X, y)
plt.plot(X, model.predict(X), color='red')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()