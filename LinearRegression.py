#Importing libraries to make it all work
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

#connecting the data to variable df
df = pd.read_csv("Salary_dataset.csv")

#creating my model
model = LinearRegression()

#connecting the columns to variables X and y
X = df[["YearsExperience"]]
y = df["Salary"]

#training the model on the data
model.fit(X, y)

predictions = model.predict(X)

plt.scatter(X, y)
plt.plot(X, predictions, color='red')
plt.xlabel("Years of experience")
plt.ylabel("Salary")
plt.show()