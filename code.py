import numpy as np
import pandas as pd
df1=pd.read_csv(r"C:\Users\spk09\Desktop\h1.csv")
from sklearn.model_selection import train_test_split

predictors = df1.drop("Target",axis=1)
target = df1["Target"]

X_train,X_test,Y_train,Y_test = train_test_split(predictors,target,test_size=0.3,random_state=0)
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf.fit(X_train,Y_train)
Y_pred_rf = rf.predict(X_test)
score_rf = round(accuracy_score(Y_pred_rf,Y_test)*100,2)
print("The accuracy score achieved using Decision Tree is: "+str(score_rf)+" %")

import pickle
with open('model3.pkl','wb') as files:
    pickle.dump(rf,files)