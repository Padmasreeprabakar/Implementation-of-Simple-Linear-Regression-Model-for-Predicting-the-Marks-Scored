# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

1.import the needed packages.

2.Assigning hours to x and scores to y.

3.Plot the scatter plot.

4.Use mse,rmse,mae formula to find the values.
```

Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SUBIKSHA K
RegisterNumber: 212224040332

```
## Program:

          # IMPORT REQUIRED PACKAGE
          import pandas as pd
          import numpy as np
          from sklearn.metrics import mean_absolute_error,mean_squared_error
          import matplotlib.pyplot as plt
          dataset=pd.read_csv('student_scores.csv')
          print(dataset)
          # READ CSV FILES
          dataset=pd.read_csv('student_scores.csv')
          print(dataset.head())
          print(dataset.tail())
          # COMPARE DATASET
          x=dataset.iloc[:,:-1].values
          print(x)
          y=dataset.iloc[:,1].values
          print(y)
          # PRINT PREDICTED VALUE
          from sklearn.model_selection import train_test_split
          x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
          from sklearn.linear_model import LinearRegression
          reg=LinearRegression()
          reg.fit(x_train,y_train)
          y_pred = reg.predict(x_test)
          print(y_pred)
          print(y_test)
          # GRAPH PLOT FOR TRAINING SET
          plt.scatter(x_train,y_train,color='purple')
          plt.plot(x_train,reg.predict(x_train),color='black')
          plt.title("Hours vs Scores(Training set)")
          plt.xlabel("Hours")
          plt.ylabel("Scores")
          plt.show()
          # GRAPH PLOT FOR TESTING SET
          plt.scatter(x_test,y_test,color='red')
          plt.plot(x_train,reg.predict(x_train),color='black')
          plt.title("Hours vs Scores(Testing set)")
          plt.xlabel("Hours")
          plt.ylabel("Scores")
          plt.show()
          # PRINT THE ERROR
          mse=mean_absolute_error(y_test,y_pred)
          print('Mean Square Error = ',mse)
          mae=mean_absolute_error(y_test,y_pred)
          print('Mean Absolute Error = ',mae)
          rmse=np.sqrt(mse)
          print("Root Mean Square Error = ",rmse)

## Output:

READ HEAD AND TAIL FILES

<img width="317" height="836" alt="Screenshot 2025-09-10 134025" src="https://github.com/user-attachments/assets/9273c5b9-a0a1-4f70-ac8c-21fa1f8115b6" />


COMPARE DATASET

<img width="724" height="590" alt="Screenshot 2025-09-10 134658" src="https://github.com/user-attachments/assets/4d66086e-50ca-4947-815c-d11011d10bb1" />

PREDICTED VALUE

<img width="720" height="65" alt="Screenshot 2025-09-10 134758" src="https://github.com/user-attachments/assets/5972162e-c44d-490c-8b7b-66f3bf10cf8c" />


GRAPH FOR TRAINING SET

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/5627d3ba-c6d0-4f39-bc71-f5c63ab507a9" />

GRAPH FOR TESTING SET

<img width="562" height="455" alt="image" src="https://github.com/user-attachments/assets/24881ec6-ac1e-4185-a571-e8279fb4fb12" />

ERROR

<img width="442" height="71" alt="Screenshot 2025-09-10 135044" src="https://github.com/user-attachments/assets/8e2bf758-e575-46cd-b658-cd66bb719894" />








## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
