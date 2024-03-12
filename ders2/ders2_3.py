import matplotlib.pyplot as plt 
import pandas as pd

veriler = pd.read_csv("veri1.csv")

x= veriler.iloc[:,1:2]
y= veriler.iloc[:,2:]

X=x.values
Y=y.values


from sklearn.tree import DecisionTreeRegressor
r_dt =DecisionTreeRegressor(random_state=0)
r_dt.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,r_dt.predict(X),color='blue')
plt.show()

print(r_dt.predict([[6]]))


import numpy as np
X_grid = np.arange(min(X),max(Y),0.01)
X_grid = X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='blue')
plt.plot(X_grid,r_dt.predict(X_grid),color='green')

plt.title("Decision Tree Regression")
plt.xlabel("Eğitim Seviyesi")
plt.ylabel("Maaş")
plt.show()
