# Machine Learning Algorithms From Scratch

* The aim of this project is to build the most common machine learning algorithms from scratch using numpy.

## Linear Regression

This is a linear approach to modelling the relationship between a scalar response and one or more explanatory variables (also known as dependent and independent variables) ([Wiki](https://en.wikipedia.org/wiki/Linear_regression))

$$
\hat{y} = b + w.x
$$
$$
\hat{y} = PredictedValue, x = Predictor, w = Weights, b = bias
$$

### Cost Function

* Mean Squared Error (mse)
$$
mse =\frac{1}{N} \sum_{i=1}^{N}(y_{i} - \hat{y_{i}})^2
$$
$$
y = ActualValue
$$

## Gradient Descent

* Change in weights
$$
dw =\frac{1}{N} \sum_{k=1}^{N}2x(\hat{y} - y)
$$
* Change in bias
$$
db =\frac{1}{N} \sum_{k=1}^{N}2(\hat{y} - y)
$$
* Update the values
$$
w = w - \alpha .dw
$$
$$
b = b - \alpha .db
$$

## Logistic Regression

This is also called Logit model and it is used to model the probability of a certain class or event existing such as male/female, win/lose, etc. Each object being detected in the image would be assigned a probability between 0 and 1 ([Wiki](https://en.wikipedia.org/wiki/Linear_regression)). 
It is similiar to linear regression but the major difference is that a sigmoid function is used to convert the values obtained from the linear model to a binary value.

### Cost/Loss Function

$$
y_{linear} = b + w.x
$$

**Sigmoid function**
$$
\hat{y} = \frac{1}{1 + e^{-y_{linear}}}
$$

* Mean Squared Error (mse)
$$
mse =\frac{1}{N} \sum_{i=1}^{N}(y_{i} - \hat{y_{i}})^2
$$

## Gradient Descent (Logit Model)

* Change in weights
$$
dw =\frac{1}{N} \sum_{k=1}^{N}2x(\hat{y} - y)
$$
* Change in bias
$$
db =\frac{1}{N} \sum_{k=1}^{N}2(\hat{y} - y)
$$
* Update the values
$$
w = w - \alpha .dw
$$
$$
b = b - \alpha .db
$$