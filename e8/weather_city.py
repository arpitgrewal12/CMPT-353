
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import sys
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.svm import SVC



#data_labelled = pd.read_csv('monthly-data-labelled.csv')
#data_unlabelled = pd.read_csv('monthly-data-unlabelled.csv')
data_labelled=pd.read_csv(sys.argv[1])
data_unlabelled=pd.read_csv(sys.argv[2])
y=data_labelled['city']
X=data_labelled[data_labelled.columns[2:62]]
Xnew=data_unlabelled[data_unlabelled.columns[2:62]]
#Xnew


X_train,X_valid,y_train,y_valid=train_test_split(X,y)

#model=VotingClassifier([
#   ('nb',GaussianNB()),
#   ('knn',KNeighborsClassifier(5)),
#   ('svm',SVC(kernel='linear',C=0.1)),
#])
#The above one 
model=make_pipeline(
    StandardScaler(),#To normalize in a predictable range
    SVC(kernel='linear',C=0.1)
    )

model.fit(X_train,y_train)
print(model.score(X_valid,y_valid))


predictions=model.predict(Xnew)

pd.Series(predictions).to_csv(sys.argv[3], index=False, header=False)
#pd.Series(predictions).to_csv('labels.csv', index=False, header=False)
#df=pd.DataFrame({'truth':y_valid,'prediction':model.predict(X_valid)})
#print(df[df['truth']!=df['prediction']])



