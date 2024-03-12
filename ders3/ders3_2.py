import matplotlib.pyplot as plt
import pandas as pd

veriler=pd.read_csv('veri1.csv')

x=veriler.iloc[:,1:2]
y=veriler.iloc[:,2:]
X=x.values
Y=y.values

from sklearn.preprocessing import StandardScaler
scaler =StandardScaler()
X_scaled=scaler.fit_transform(X)

from sklearn.svm import SVR
svr_reg =SVR(kermnel='poly') # 'linear','poly','sigmoid','rbf'
svr_reg.fit(X_scaled,Y)

plt.scatter(X,Y,color='red')
plt.plot(X,svr_reg.predict(X_scaled),color='blue')
plt.show()

print(svr_reg.predict([[6.6]]))

import numpy as np
X_grid=np.arange(min(X),max(X),0.01)
X_grid=X_grid.reshape(len(X_grid),1)

plt.scatter(X,y,color='blue')
plt.plot(X_grid,svr_reg.predict(X_grid),color='green')

plt.title('SVR REGRESSİON')
plt.xlabel('Eğitim Seviyesi')
plt.ylabel('Maaş')
plt.show()