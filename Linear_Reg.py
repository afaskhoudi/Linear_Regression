import numpy as np
from sklearn.linear_model import LinearRegression

x=np.array([9,24,31,47,51,60]).reshape((-1,1))
y=np.array([4,15,10,32,24,45])
model=LinearRegression()
model.fit(x,y)
regressor=model.score(x,y)
predict=model.predict(x)
print(f"coefficient of determination: {regressor}")
print(f"intercept: {model.intercept_}")
print(f"slope: {model.coef_}")
print(f"predicted response: {predict}")
predict=model.intercept_ + model.coef_ * x
print(f"predicted response: {predict.reshape((-1,1))}")
x_=np.arange(5).reshape((-1,1))
print(f"x_: {x_}")
y_=model.predict(x_)
print(f"y_: {y_.reshape((-1,1))}")