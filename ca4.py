import csv
import pandas as pd
import math
import numpy as np
import datetime
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder


def preprocessDate(df)  :
    df['year'] = df['Date'].dt.year
    df['month'] = df['Date'].dt.month
    df['day'] = df['Date'].dt.day
    df['dayOfWeek'] = df['Date'].dt.dayofweek
    df['is_weekend'] = np.where(df['Date'].dt.day_name().isin(['Sunday','Saturday']),1,0)
    del df['Date']
    return df

def preprocessLabeledCategorical(data, df) : 
    countries = df['Country'].tolist() 
    le = preprocessing.LabelEncoder()
    le.fit(countries)
    # print(list(le.classes_))
    transformedCountries = le.transform(countries)
    for i in range(len(data)) : 
        data[i]['Country'] = transformedCountries[i]
    
    return data

def getData(dataFile, encodingType) : 
    data = []
    col_list = ["Customer ID", "Total Quantity", "Total Price", "Country", "Date", "Is Back", "Purchase Count"]
    df = pd.read_csv(dataFile, usecols=col_list)
    df['Date']= pd.to_datetime(df['Date'])
    
    if (encodingType == "oneHot") :
        df = pd.concat([df,pd.get_dummies(df['Country'])],axis=1).drop(['Country'],axis=1)
    
    df = preprocessDate(df)
    
    for index, dfRow in df.iterrows() : 
        data.append(dfRow)
    
    if (encodingType == "labeled") :
        preprocessLabeledCategorical(data, df)
    
    return data

    
data = getData("data.csv", "oneHot")

print(data[0])
print(data[1])
print(data[2])
print(data[3])
print(data[4])
print(data[5])
