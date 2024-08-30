# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
STEP 1 : Start

STEP 2 : Import 'pandas', 'numpy', 'matplotlib', and 'sklearn' for data processing, visualization, and machine learning.

STEP 3 : Read the student scores dataset from a CSV file. 

STEP 4 : Separate the independent variable (Hours) into 'X' and the dependent variable (Scores) into 'Y'.

STEP 5 : Divide the dataset into training and testing sets using 'train_test_split'. 

STEP 6 : Create a LinearRegression model and train it with the training data ('X_train', 'Y_train').

STEP 7 : Predict the scores for the test data ('X_test') using the trained model.

STEP 8 : Plot the training data with the regression line and separately plot the test data with the regression line.

STEP 9 : Calculate and print the Mean Squared Error (MSE), Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE) to assess model accuracy.

STEP 10 : Stop

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: MANOJ MV
RegisterNumber:  212222220023
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv("C:/Users/SEC/Downloads/student_scores.csv")
df.head()

df.tail()

#segregating data to variables
X=df.iloc[:,:-1].values
X

Y=df.iloc[:,1].values
Y

#splitting training and test data
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=1/3,random_state=0)

from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(X_train,Y_train)
Y_pred=regressor.predict(X_test)

#displaying predicted values
Y_pred

Y_test

#graph plot for training data
plt.scatter(X_train,Y_train,color="orange")
plt.plot(X_train,regressor.predict(X_train),color="red")
plt.title("Hours vs Scores(Training Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

plt.scatter(X_train,Y_train,color="purple")
plt.plot(X_test,regressor.predict(X_test),color="yellow")
plt.title("Hours vs Scores(Test Set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()

mse=mean_squared_error(Y_test,Y_pred)
print('MSE = ',mse)

mae=mean_absolute_error(Y_test,Y_pred)
print('MAE = ',mae)

rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```

## Output:
## Y_Pred
![image](https://github.com/user-attachments/assets/3d861ba3-d16a-4773-a6f1-6db88091cbcb)

## Y_test
![image](https://github.com/user-attachments/assets/db50ad83-7132-4566-96d9-99ea609bfaab)

## Training Set
![image](https://github.com/user-attachments/assets/d037ef43-e9f9-4560-bba5-6a72d7f73abe)

## Testing Set
![image](https://github.com/user-attachments/assets/cbc88e0e-7c7f-4ade-bf65-0d83aa204d47)

## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
