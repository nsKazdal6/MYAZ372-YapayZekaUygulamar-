import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv('veri1.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X=x.values
Y=y.values

from sklearn.neighbors import KNeighborsRegressor

knn_reg =KNeighborsRegressor(n_neighbors=2)
knn_reg.fit(X,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,knn_reg.predict(X),color='blue')
plt.show()

print(knn_reg.predict([[6.6]]))

import numpy as np
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='blue')
plt.plot(X_grid,knn_reg.predict(X_grid),color='green')

plt.title('KNN REGRESSİON')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.show()