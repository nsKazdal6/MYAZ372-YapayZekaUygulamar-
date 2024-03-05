import matplotlib.pyplot as plt 
import pandas as pd

veriler = pd.read_csv("veri1.csv")

x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

X=x.values
Y=y.values

from sklearn.linear_model import LinearRegression
lin_reg=LinearRegression()
lin_reg.fit(X,Y)



from sklearn.preprocessing import PolynomialFeatures
poly_reg2 = PolynomialFeatures(degree=2)
x_poly2 =poly_reg2.fit_transform(X)
lin_reg2 =LinearRegression()
lin_reg2.fit(x_poly2,y)

poly_reg3 = PolynomialFeatures(degree=4)
x_poly3 =poly_reg3.fit_transform(X)
lin_reg3 =LinearRegression()
lin_reg3.fit(x_poly3,y)



plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg.predict(X),color='blue')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg2.predict(poly_reg2.fit_transform(X)),color='yellow')
plt.show()

plt.scatter(X,Y,color='red')
plt.plot(x,lin_reg3.predict(poly_reg3.fit_transform(X)),color='purple')
plt.show()





print(lin_reg.predict([[6.6]]))
print(lin_reg2.predict(poly_reg2.fit_transform([[6.6]])))
