# Basics
import numpy as np
import pandas as pd
import joblib

# For preprocessing
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV

# for machine learning modelling
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.metrics import roc_auc_score, confusion_matrix

# feature importance
import shap


app_train = pd.read_csv('data/application_train_cleaned.csv').drop(columns='Unnamed: 0')
app_test = pd.read_csv('data/application_test_cleaned.csv').drop(columns='Unnamed: 0')

X_train = app_train.drop(columns=['TARGET','SK_ID_CURR'])
y_train = app_train['TARGET']

rf = RandomForestClassifier(max_features= 'sqrt', criterion ='gini', n_estimators=70, max_depth=15, random_state=42)
rf2 = RandomForestClassifier(max_features= 'sqrt', criterion ='gini',n_estimators=100, max_depth=15, random_state=5)
rf3 = RandomForestClassifier(max_features= 'sqrt', criterion ='gini', n_estimators=75, max_depth=15, random_state=30)
rf4 = RandomForestClassifier(max_features= 'sqrt', criterion ='gini', n_estimators=80, max_depth=15, random_state=4)
rf5 = RandomForestClassifier(max_features= 'sqrt', criterion ='gini', n_estimators=100, max_depth=15, random_state=12)

rf.fit(X_train, y_train)
rf2.fit(X_train, y_train)
rf3.fit(X_train, y_train)
rf4.fit(X_train, y_train)
rf5.fit(X_train, y_train)

filename = 'rf.sav'
filename2 = 'rf2.sav'
filename3 = 'rf3.sav'
filename4 = 'rf4.sav'
filename5 = 'rf5.sav'

joblib.dump(rf, filename)
joblib.dump(rf2, filename2)
joblib.dump(rf3, filename3)
joblib.dump(rf4, filename4)
joblib.dump(rf5, filename5)

print('dumping done')