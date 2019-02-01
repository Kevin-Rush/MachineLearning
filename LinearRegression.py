#equation of a line is y=mx + b 
# m = slope
# b = y-int
# m = ((MOA) x * (MOA) y - (MOA) x*y) / (((MOA) x )^2 - (MOA) x^2)
# b = (MOA) y - m*(MOA)x
 
from statistics import mean 
import numpy as np
import matplotlib.pyplot as plt 
from matplotlib import style 

style.use('fivethirtyeight')

xs = np.array([1,2,3,4,5,6], dtype=np.float64)
ys = np.array([5,4,6,5,6,7], dtype=np.float64)

def best_fit_slope_and_int(xs, ys):
    m = ((mean(xs) * mean(ys)) - mean(xs*ys))/((mean(xs))**2 - mean(xs**2))
    b = mean(ys) - m*mean(xs)
    return m, b

m, b = best_fit_slope_and_int(xs, ys)

#print(m)
#print(b)

regression_line = [(m*x)+b for x in xs]

#print(regression_line)

predict_x = 8
predict_y = m*predict_x + b
#we have been able to accumulate data, find a best fit line based on that data (our model) so we can now make predictions based on that data
#obviously the more data we have the better our model of the data is and the better our perdiction can be
#this works but even though we have a best fit line we also want it to be a good fit
#how accurate is our best fit line?? we need to calculate how good our best fit line i
#in linear regression accuracy and confidence are pretty similair until we go farther into linear regression
plt.scatter(xs, ys)
plt.scatter(predict_x,predict_y, color='g')
plt.plot(xs, regression_line)
plt.show()