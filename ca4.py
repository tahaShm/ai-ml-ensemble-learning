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

def preprocessCategorical(data, encodingType, df) : 
    # print(uniqueCountries)
    # enc = OneHotEncoder(handle_unknown='ignore')
    # X = [['Male'], ['Female'], ['Female']]
    countries = df['Country'].tolist()
    # print(countries) : 
    if (encodingType == "labeled") : 
        le = preprocessing.LabelEncoder()
        le.fit(countries)
        # print(list(le.classes_))
        transformedCountries = le.transform(countries)
        for i in range(len(data)) : 
            data[i]['Country'] = transformedCountries[i]
            
    elif (encodingType == "oneHot") : 
        enc = OneHotEncoder(handle_unknown='ignore')
        for i in range(len(countries)) : 
            countries[i] = [countries[i]]
        enc.fit(countries)
        # print(enc.categories_)

        transformedCountries = enc.transform(countries).toarray()
        for i in range(len(data)) : 
            data[i][countries[i][0]] = transformedCountries[i]


    # enc.inverse_transform([[0, 1, 1, 0, 0], [0, 0, 0, 1, 0]])


    # print(enc.get_feature_names(['gender', 'group']))
    return data

def getData(dataFile) : 
    data = []
    col_list = ["Customer ID", "Total Quantity", "Total Price", "Country", "Date", "Is Back", "Purchase Count"]
    df = pd.read_csv(dataFile, usecols=col_list)
    df['Date']= pd.to_datetime(df['Date']) 
    
    df = preprocessDate(df)
    
    for index, dfRow in df.iterrows() : 
        data.append(dfRow)
    
    preprocessCategorical(data, "oneHot", df)
    # data = preprocessCategorical("labeled")
    
    return data

    
data = getData("data.csv")

# print(data[0])
# print(data[1])
# print(data[2])
# print(data[3])
# print(data[4])
# print(data[5])
