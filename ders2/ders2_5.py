import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error ,mean_squared_error,r2_score

veriler =pd.read_csv("arac_verileri.csv",sep=";")

x=veriler[['GP','DP','EP','MRGDPI','TNL','TREP']]
y=veriler[['BEV']]


Y=y.replace({',':'.'},regex=True)
X=x.replace({',':'.'},regex=True)

from sklearn.model_selection import train_test_split
x_train, x_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=42)



from sklearn.linear_model import Lasso,LinearRegression
lr=LinearRegression()
lr.fit(x_train,y_train)

y_perd = lr.predict(x_test)

r2=r2_score(y_test,y_perd)
mse=mean_squared_error(y_test,y_perd)

print("R-squared(R2):",r2)
print("MSE:",mse)
