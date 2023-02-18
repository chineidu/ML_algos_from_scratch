# Machine Learning Algorithms From Scratch

* The aim of this project is to build the most common machine learning algorithms from scratch using numpy.

## Linear Regression

This is a linear approach to modelling the relationship between a scalar response (dependent variable) and one or more explanatory (independent) variables. ([Wiki](https://en.wikipedia.org/wiki/Linear_regression))

### Cost Function

* Mean Squared Error (mse) is minimized in this algorithm using gradient descent.
* *dw* and *db* are the change in weights and bias respectively.

[![image.png](https://i.postimg.cc/SKJGNBQ9/image.png)](https://postimg.cc/WDcZ5fkp)

<hr>

[![image.png](https://i.postimg.cc/TY5GgwkN/image.png)](https://postimg.cc/jLTGrsYy)

## Logistic Regression

This is also called **Logit model** and it is used to model the probability of a certain class or event existing such as male/female, win/lose, etc. Each object being detected in the image would be assigned a probability between 0 and 1 ([Wiki](https://en.wikipedia.org/wiki/Linear_regression)).
It is similiar to linear regression but the major difference is that a **sigmoid function** is used to convert the values obtained from the linear model to values between 0 and 1.
![logistic](https://i.postimg.cc/2yFR6bjV/image.png)

* The weights and bias are frequently updated until the best/minimum values are obtained.
