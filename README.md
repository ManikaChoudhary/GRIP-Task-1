# GRIP-Task-1
#Importing all the required libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

#Reading data of csv file from the given url
df = pd.read_csv("http://bit.ly/w-data")
df.head(10)

#This function returns the number of column and rows of the data framework
df.shape

#Checking if there are any null values
df.isnull().any()

#Plotting the scores
sns.scatterplot(x='Hours' ,y='Scores',data=df)

#Dividing the data into attributes (inputs) and labels (outputs)
X = df.iloc[:,:-1]
Y = df.iloc[:,1]

X.head()

#Splitting the data into training and tests sets by using Scikit-Learn’s built-in train_test_split() method:
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=0)

#Splitting the independent and dependent features into train test split using sklearn library
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size=0.2, random_state=42, shuffle=True)

#Training the algorithm 
from sklearn.linear_model import LinearRegression
lr = LinearRegression()

lr.fit(X_train,Y_train)

#Predection of test data
Y_pred = lr.predict(X_test)

#Plotting the regression line
line = lr.coef_*X+lr.intercept_

#Plotting of the test data
plt.scatter(X, Y)
plt.plot(X, line);
plt.show()

#Evaluation of the model
from sklearn.metrics import mean_absolute_error,mean_squared_error
print("Mean_absolute_error is" ,mean_absolute_error(Y_test,Y_pred))
print("mean_squared_error is" , mean_squared_error(Y_test,Y_pred))

#Comparison of predicted and actual values
pred_actual = pd.DataFrame({'Actual': Y_test , 'Predicted' : Y_pred})
pred_actual

from sklearn.metrics import r2_score
score = r2_score(Y_test,Y_pred)
score


print("If a student studies for 9.25hrs/day then he will score approximately {}".format(lr.predict([[9.25]])))

print("If a student studies for 9.5hrs/day then he will approximately score 92.38611528")
