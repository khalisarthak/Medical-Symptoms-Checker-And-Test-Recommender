import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics
import joblib
df=pd.read_csv('pre_processed.csv')
X=df.drop(['Disease'], axis=1)
y=df.iloc[:,-1]
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3,random_state=42)
forest=RandomForestClassifier(n_estimators=30)
forest.fit(X_train,y_train)
pred=forest.predict(X_test)
joblib.dump(forest,"random_forest.joblib")