# Basics
import numpy as np
import pandas as pd
import joblib

# for pre-processing
from sklearn.preprocessing import OrdinalEncoder
from category_encoders import TargetEncoder 
from sklearn.model_selection import train_test_split

def delete_sparse_columns(df):
    df1 = df.drop(df.columns[df.isnull().mean()>0.4],axis=1, inplace=True)
    return df1

def delete_unintelligible_columns(df):
    unintelligible_columns = []
    for i in range(2,22):
        unintelligible_columns.append(f"FLAG_DOCUMENT_{i}")
    unintelligible_columns.append('EXT_SOURCE_2')
    unintelligible_columns.append('EXT_SOURCE_3')
    df1 = df.drop(columns=unintelligible_columns, inplace=True)
    return df1

def delete_discriminatory_columns(df):
    df1 = df.drop(columns='CODE_GENDER', inplace=True)
    return df1

def delete_columns_due_to_correlation(df):
    columns_to_remove = ['CNT_FAM_MEMBERS', 'REGION_RATING_CLIENT', 'LIVE_REGION_NOT_WORK_REGION', 'LIVE_CITY_NOT_WORK_CITY', 'OBS_60_CNT_SOCIAL_CIRCLE', 'DEF_60_CNT_SOCIAL_CIRCLE']
    df1 = df.drop(columns=columns_to_remove, inplace=True)
    return df1

def impute_low_nan_numbers_columns(train, test):
    cat_columns_lower_percentage_nan  = [i for i in train.columns[(((train.isnull().sum() / train.shape[0]) * 100) > 0) 
                                                                  & (((train.isnull().sum() / train.shape[0]) * 100) < 14)] 
                                     if train[i].dtype == 'O']
    num_columns_lower_percentage_nan  = [i for i in train.columns[(((train.isnull().sum() / train.shape[0]) * 100) > 0) 
                                                                  & (((train.isnull().sum() / train.shape[0]) * 100) < 14)] 
                                     if train[i].dtype != 'O']
    cat_columns_train_mode = []
    for i in cat_columns_lower_percentage_nan:
        test[i].fillna(train[i].mode()[0], inplace=True)
        train[i].fillna(train[i].mode()[0], inplace=True)
    col_mod_transform = [i for i in num_columns_lower_percentage_nan if i not in ['AMT_ANNUITY','AMT_GOODS_PRICE']]
    for i in col_mod_transform:
        test[i].fillna(train[i].mode()[0], inplace=True)
        train[i].fillna(train[i].mode()[0], inplace=True)
    test['AMT_ANNUITY'].fillna(train['AMT_ANNUITY'].mean(), inplace=True)
    train['AMT_ANNUITY'].fillna(train['AMT_ANNUITY'].mean(), inplace=True)
    test['AMT_GOODS_PRICE'].fillna(train['AMT_GOODS_PRICE'].median(), inplace = True)
    train['AMT_GOODS_PRICE'].fillna(train['AMT_GOODS_PRICE'].median(), inplace = True)
    return train, test

def impute_occupation_type(train, test):
    for education in train['NAME_EDUCATION_TYPE'].unique():
        occupation_value = train[train['NAME_EDUCATION_TYPE'] == education]['OCCUPATION_TYPE'].mode()[0]
        for df in [train, test]:
            mask = (df['NAME_EDUCATION_TYPE'] == education) & (df['OCCUPATION_TYPE'].isna())
            df.loc[mask, 'OCCUPATION_TYPE'] = occupation_value
    return train, test

def impute_outliers(df):
    all_numerical_cols = list(df.select_dtypes(exclude='object').columns)
    cont_cols = [col for col in all_numerical_cols if col != "TARGET" and col != 'SK_ID_CURR']
    
    for col in cont_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        IQR = q3-q1
        upper = q3 + 1.5 * IQR
        lower = q1 - 1.5 * IQR
        
        mask_upper = (df[col] > upper) 
        mask_lower = (df[col] < lower)
        
        df.loc[mask_upper, col] = upper 
        df.loc[mask_lower, col] = lower
        
    return df

def modify_categorical_values(train, test):
    test['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'}, inplace=True)
    train['NAME_TYPE_SUITE'].replace({'Other_A':'Other','Other_B':'Other','Group of people':'Other'}, inplace=True)
    test['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'}, inplace=True)
    train['NAME_INCOME_TYPE'].replace({'Unemployed':'Other','Student':'Other','Maternity leave':'Other'}, inplace=True)
    
    others = train['ORGANIZATION_TYPE'].value_counts().index[15:]
    label = 'Other'
    train['ORGANIZATION_TYPE'].replace(others, label, inplace=True)
    test['ORGANIZATION_TYPE'].replace(others, label, inplace=True)
    
    return train, test

def feature_engineering(df):
    df['LTV'] = df['AMT_CREDIT']/df['AMT_GOODS_PRICE']
    df['DTI'] = df['AMT_ANNUITY']/df['AMT_INCOME_TOTAL']
    df['Employed/Birth'] = df['DAYS_EMPLOYED']/df['DAYS_BIRTH'] 
    df['Flag_Greater_30'] = (df['DAYS_BIRTH']/-365.25).apply(lambda x: 1 if x > 30 else 0)
    df['Flag_Employment_Greater_5'] = (df['DAYS_EMPLOYED']/-365.25).apply(lambda x: 1 if x > 5 else 0)
    return df

def encoder(train, test):
    cat_col = train.select_dtypes('object')
    enc = TargetEncoder()
    train[cat_col.columns] = enc.fit_transform(train[cat_col.columns], train['TARGET'])
    test[cat_col.columns] = enc.transform(test[cat_col.columns]) 
    return train, test

def data_cleaning(train, test):
	delete_sparse_columns(train)
	delete_sparse_columns(test)
	delete_unintelligible_columns(train)
	delete_unintelligible_columns(test)
	delete_discriminatory_columns(train)
	delete_discriminatory_columns(test)
	delete_columns_due_to_correlation(train)
	delete_columns_due_to_correlation(test)
	impute_low_nan_numbers_columns(train, test)
	impute_occupation_type(train, test)
	impute_outliers(train)
	impute_outliers(test)
	modify_categorical_values(train, test)
	encoder(train, test)
	feature_engineering(train)
	feature_engineering(test)
	return train, test

def get_refunding_proba(sk_id_curr, data):
	rf = joblib.load('rf.sav')
	rf2 = joblib.load('rf2.sav')
	rf3 = joblib.load('rf3.sav')
	rf4 = joblib.load('rf4.sav')
	rf5 = joblib.load('rf5.sav')
	proba1 = rf.predict_proba(data.loc[data['SK_ID_CURR'] == sk_id_curr].drop(columns='SK_ID_CURR'))[0][0]
	proba2 = rf2.predict_proba(data.loc[data['SK_ID_CURR'] == sk_id_curr].drop(columns='SK_ID_CURR'))[0][0]
	proba3 = rf3.predict_proba(data.loc[data['SK_ID_CURR'] == sk_id_curr].drop(columns='SK_ID_CURR'))[0][0]
	proba4 = rf4.predict_proba(data.loc[data['SK_ID_CURR'] == sk_id_curr].drop(columns='SK_ID_CURR'))[0][0]
	proba5 = rf5.predict_proba(data.loc[data['SK_ID_CURR'] == sk_id_curr].drop(columns='SK_ID_CURR'))[0][0]
	proba = (proba1 + proba2 + proba3 + proba4 + proba5)/5
	if proba >=0.7:
		answer = 'loan granted'
	else:
		answer = 'loan denied'
	return proba, answer
