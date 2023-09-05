# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

# AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

# Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

# Algorithm

1.Import the standard Libraries.
2.Set variables for assigning dataset values
3.Import linear regression from sklearn
4.Assign the points for representing in the graph
5.Predict the regression for marks by using the representation of the graph. 
6.Compare the graphs and hence we obtained the linear regression for the given datas

# Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: Suji.G
RegisterNumber: 212222230152

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
df.head()
df.tail()
x = df.iloc[:,:-1].values
x
y = df.iloc[:,1].values
y
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train,y_train)
y_pred = regressor.predict(x_test)
y_pred
y_test
#Graph plot for training data
plt.scatter(x_train,y_train,color='black')
plt.plot(x_train,regressor.predict(x_train),color='purple')
plt.title("Hours vs Scores(Training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
#Graph plot for test data
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Hours vs Scores(Testing set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_absolute_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print("RMSE= ",rmse)
```

# Output:
### df.head()
![Screenshot 2023-09-05 104348](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/b153bb27-164d-43de-b78f-2d687047915e)

### df.tail()
![Screenshot 2023-09-05 104810](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/cc40deec-122c-41f7-a1dd-99c7120ac978)

### Array value of X
![Screenshot 2023-09-05 104848](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/203ddaa2-3258-4d60-b305-20aad1232536)

### Array value of Y
![Screenshot 2023-09-05 104950](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/aa4401bb-281b-44bf-892b-493f082afb05)

### Array values of Y test
![Screenshot 2023-09-05 105035](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/c4366894-7578-41ad-ab91-1bc3f3fa27d4)

### Training Set Graph
![Screenshot 2023-09-05 105157](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/543d3531-59cc-401b-8e6e-85d2b938c49a)

### Test Set Graph
![Screenshot 2023-09-05 105210](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/b9242df8-4530-41e4-9819-76840f3ca757)

### Values of MSE, MAE and RMSE
![Screenshot 2023-09-05 105218](https://github.com/sujigunasekar/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/119559822/6f20014e-2e8b-4218-9ffd-f6f174d43a7d)

# Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
