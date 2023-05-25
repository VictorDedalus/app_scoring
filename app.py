# Basics
import numpy as np
import pandas as pd
import joblib

from fonctions import get_refunding_proba

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier

columns_description = pd.read_csv('columns_description.csv').drop(columns='Unnamed: 0')

print(columns_description)
