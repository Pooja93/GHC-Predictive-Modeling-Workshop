
# coding: utf-8

# In[3]:

# loading imdb data into a python list format

import csv

imdb_data_csv= csv.reader(open('movie_metadata.csv'))
imdb_data=[]
for item in imdb_data_csv:
    imdb_data.append(item)
# step 1: preprocessing
# remove NAN values from the data
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import OneHotEncoder
def remove_NAN(data):
     imp=Imputer(missing_values='nan',strategy="mean",axis=0)
     imp.fit_transform(data)
     return data
#one hot encoding
def convert_text_to_numeric_onehot(data,column_nos):
    #enc = OneHotEncoder()
    #enc.fit([[0, 0, 3], [1, 1, 0], [0, 2, 1], [1, 0, 2]])  
    enc=OneHotEncoder(categorical_features=column_nos)
    enc.fit(data);
    return enc.transform(data);
    #array([[ 1.,  0.,  0.,  1.,  0.,  0.,  0.,  0.,  1.]])

# for regression we would ideally like all the fields to be numeric, therefore, convert names and other text fields into numbers
def convert_text_to_numeric(data,column_no):
    encoding={}
    curr_index = 1
    for i,item in enumerate(data):
        if item[column_no] in encoding.keys():
            data[i][column_no] = encoding[item[column_no]]
        else:
            encoding[item[column_no]] = curr_index
            data[i][column_no] = curr_index
            curr_index+=1
    return data

data = imdb_data[1:]
data = convert_text_to_numeric(data,1)
data = convert_text_to_numeric(data,6);data = convert_text_to_numeric(data,10);data = convert_text_to_numeric(data,14);
data = convert_text_to_numeric(data,19);data = convert_text_to_numeric(data,20);data = convert_text_to_numeric(data,21)

# remove some columns from the data
for row in data:
   del row[17]
   del row[16]
   del row[11]
   del row[9]
   del row[0]

import numpy as np
import ast
def toFloat(data):
    data1=[]
    for i,item in enumerate(data):
        li = []
        for x in item:
            if isinstance(x, str):
                try:	
                    li.append(float(x) if '.' in x else int(x))
                except:
                    li.append(0.0)
            else:
                li.append(x)
        data1.append(li)
    return data1

data_float = toFloat(data)
data_np = np.matrix(data_float)
cat_features=[0,5,8,11,14,15,16];
data_np_x = np.delete(data_np, [20], axis=1)
data_label=data_np[:,20];
data_np_onehot = convert_text_to_numeric_onehot(data_np_x,cat_features)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(data_np_onehot, data_label, 
                                                              test_size=0.25, random_state=0)
# apply regression and voila !!
from sklearn import linear_model

lr = linear_model.LinearRegression()
lr = linear_model.Ridge()
lr.fit(x_train, y_train)
y_pred = lr.predict(x_test)

# model evaluation
from sklearn.metrics import mean_absolute_error
print 'linear regression absolute error: ', mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print 'linear regression squared error: ',mean_squared_error(y_test, y_pred)
#exit(0)

from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
regr_ridge = Ridge(alpha=10);
regr_ridge.fit(x_train, y_train)
y_pred = regr_ridge.predict(x_test)
# model evaluation
from sklearn.metrics import mean_absolute_error
print 'ridge absolute error: ', mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print 'ridge squared error: ',mean_squared_error(y_test, y_pred)

# In[28]:

# using svm regression
from sklearn.svm import SVR
regr_svr = SVR(C=0.5, epsilon=0.2)
regr_svr.fit(x_train, y_train)
y_pred = regr_svr.predict(x_test)

# model evaluation
from sklearn.metrics import mean_absolute_error
print 'SVR absolute error: ', mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print 'SVR squared error: ',mean_squared_error(y_test, y_pred)


# In[29]:
from sklearn.linear_model import Lasso
regr_ls = Lasso(normalize=True, max_iter=1e5,alpha=1.0)
regr_ls.fit(x_train, y_train)
y_pred = regr_ls.predict(x_test)
# model evaluation
from sklearn.metrics import mean_absolute_error
print 'Lasso absolute error: ', mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print 'Lasso squared error: ',mean_squared_error(y_test, y_pred)

from sklearn.ensemble import RandomForestRegressor
regr_rf = RandomForestRegressor(max_depth=2);
regr_rf.fit(x_train, y_train)
y_pred = regr_rf.predict(x_test)

# model evaluation
from sklearn.metrics import mean_absolute_error
print 'Random Forest absolute error: ', mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print 'Random Forest squared error: ',mean_squared_error(y_test, y_pred)

from sklearn.tree import DecisionTreeRegressor
regr_dt = DecisionTreeRegressor(max_depth=2)
regr_dt.fit(x_train, y_train)
y_pred = regr_dt.predict(x_test)

# model evaluation
from sklearn.metrics import mean_absolute_error
print 'decision tree absolute error: ', mean_absolute_error(y_test, y_pred)

from sklearn.metrics import mean_squared_error
print 'decsion tree squared error: ',mean_squared_error(y_test, y_pred)

print "You have successfully executed the test code !! Hope to see you at the workshop !!"



