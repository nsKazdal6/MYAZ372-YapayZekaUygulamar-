import pandas as pd



df = pd.read_csv("linear_regression_dataset.csv",sep=";")

x=df.deneyim.values.reshape(-1,1)
y=df.maas.values.reshape(-1,1)

from sklearn.linear_model import LinearRegression
xxx=LinearRegression()
xxx.fit(x,y)

print(xxx.predict([[50]]))