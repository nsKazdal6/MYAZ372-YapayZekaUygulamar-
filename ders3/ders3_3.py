import pandas as pd 
dataset=pd.read_csv(r'C:\Users\nazar\OneDrive\Desktop\YapayZekaUygulamaları\ders3\data_satınalma.csv')

X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,
                                                 test_size=0.2,
                                                 random_state=42)

from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train,y_train)
print("test accurac {}".format(classifier.score(X_test,y_test)*100))

y_pred=classifier.predict(X_test)
print((y_test==y_pred).mean())

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
print(cm)
