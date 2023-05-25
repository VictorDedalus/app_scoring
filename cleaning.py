# Basics
import numpy as np
import pandas as pd

from fonctions import data_cleaning

app_train = pd.read_csv('data/application_train.csv')
app_test = pd.read_csv('data/application_test.csv')

app_train, app_test = data_cleaning(app_train, app_test)

app_train.to_csv('data/application_train_cleaned.csv')
app_test.to_csv('data/application_test_cleaned.csv')

columns_description=pd.read_csv('HomeCredit_columns_description.csv', encoding = "ISO-8859-1").drop(columns=['Unnamed: 0', 'Table', 'Special'])

LTV = {'Row': 'LTV', 'Description': 'Loan to value ratio'}
DTI = {'Row': 'DTI','Description': 'Debt to income ratio'}
employed_birth  = {'Row':'Employed/Birth', 'Description': 'Days employed percentage'}
age_above_30 ={'Row': 'Flag_Greater_30', 'Description': 'Is the client older than 30 years old'}
employment_above_5_years = {'Row': 'Flag_Employment_Greater_5', 'Description': 'Has the client been employed for longer than 5 years'}

columns_description = columns_description.append(LTV, ignore_index=True).append(DTI, ignore_index=True).append(employed_birth, ignore_index=True).append(age_above_30, ignore_index=True).append(employment_above_5_years, ignore_index=True)

columns_description.to_csv('columns_description.csv')