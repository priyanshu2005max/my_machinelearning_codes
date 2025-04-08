# experiment 1(ml)
#import all the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
#STEP 2
diamonds=sns.load_dataset('diamonds')
#STEP 3
print(diamonds.head())
print(diamonds.info())
print(diamonds.shape)
#STEP 4
features=['carat','depth','table','x','y','z']
x=diamonds[features]
y=diamonds['price']
#STEP 5
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#STEP 6
#print(X_train)
print(y_train)
#print(X_test)
#print(y_test)
#STEP 7
model=LinearRegression()
model.fit(X_train,y_train)
#STEP 8
y_pred=model.predict(X_test)
#STEP 9
print("predicted value",y_pred)
print("actual value",y_test)
#STEP 10
from sklearn.metrics import mean_squared_error, r2_score # Import necessary functions

mse = mean_squared_error(y_test, y_pred)  # Calculate mse
r2 = r2_score(y_test, y_pred)
#STEP 11
print("mean squared error",mse)
print("r - squared",r2)
#STEP 12
plt.figure(figsize=(10,6))
plt.scatter(y_test,y_pred)
plt.plot(y_test,y_test,'r')
plt.xlabel("actual price")
plt.ylabel("predicted price")
plt.title("actual vs predicted price")
plt.grid(True)
plt.show()












