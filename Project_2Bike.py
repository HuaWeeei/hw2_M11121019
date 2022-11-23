# -*- coding: utf-8 -*-
"""
Created on Thu Nov 17 18:03:20 2022

@author: bbill
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import time
import xgboost as xgb

    
data=pd.read_csv("bike_train.csv")
data_t=pd.read_csv("bike_test.csv")
#匯入X


for i in range(len(data['Visibility (10m)'])):
    if int(data['Visibility (10m)'][i]) <=100 :
        data['2'][i] =0
    elif int(data['Visibility (10m)'][i]) >=101 and int(data['Visibility (10m)'][i]) <= 500:
        data['2'][i] =1
    elif int(data['Visibility (10m)'][i]) >=501 and int(data['Visibility (10m)'][i]) <= 1000:
        data['2'][i] = 2 
    elif int(data['Visibility (10m)'][i]) >=1001 and int(data['Visibility (10m)'][i]) <= 1500:
        data['2'][i] = 3 
    else:
        data['2'][i] = 4 
i=0
for i in range(len(data_t['Visibility (10m)'])):
    if int(data_t['Visibility (10m)'][i]) <=100 :
        data_t['2'][i] =0
    elif int(data_t['Visibility (10m)'][i]) >=101 and int(data_t['Visibility (10m)'][i]) <= 500:
        data_t['2'][i] =1
    elif int(data_t['Visibility (10m)'][i]) >=501 and int(data_t['Visibility (10m)'][i]) <= 1000:
        data_t['2'][i] = 2 
    elif int(data_t['Visibility (10m)'][i]) >=1001 and int(data_t['Visibility (10m)'][i]) <= 1500:
        data_t['2'][i] = 3 
    else:
        data_t['2'][i] = 4 

X = data.iloc[:,2:15]
y= data['Rented Bike Count']



#類別化(把文字改數字)
labelencoder = LabelEncoder() 
X['Seasons']= labelencoder.fit_transform(X['Seasons'])
X['Holiday']= labelencoder.fit_transform(X['Holiday'])
X['Functioning Day']= labelencoder.fit_transform(X['Functioning Day'])
#刪除不必要行
# X = X.drop("Holiday",axis=1)
# X = X.drop("Snowfall (cm)",axis=1)
X = X.drop("Visibility (10m)",axis=1) #重複

# #標準化X
X_s = StandardScaler()       
X = X_s.fit_transform(X)
Y = data.iloc[:,1:2]
y_s = StandardScaler()
y = y_s.fit_transform(Y)
train_X, test_X, train_y, test_y = train_test_split(X,y,test_size=0.2,random_state=(42))

cv=5

# # 建立 XGBRegressor 模型
xgbrModel=xgb.XGBRegressor()
cv_params = {'max_depth':np.linspace(2,10,5,dtype=int)
            ,'n_estimators':np.linspace(500,1000,5,dtype=int),'eta':[0.1,0.05,0.01]}
tStart_xgm = time.time()
gs_m = GridSearchCV(xgbrModel , cv_params, verbose=2,refit=True, n_jobs=1 ,cv=cv)
gs_m.fit(train_X,train_y)
gs_best = gs_m.best_estimator_
import matplotlib.pyplot as plt
importancesxgb=gs_best.feature_importances_
for i,v in enumerate(importancesxgb):
 print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importancesxgb))], importancesxgb)
plt.show()
XGBp = gs_best.predict(test_X)
tEnd_xgm = time.time()  
XGBp = np.array(XGBp).reshape(-1,1)
XGBp_r = y_s.inverse_transform(XGBp)

from sklearn import metrics
def rmse(oar,par):
    return metrics.mean_squared_error(oar,par)**0.5
def mae(oar,par):
    return metrics.mean_absolute_error(oar,par)
def mape(oar,par):
    return metrics.mean_absolute_percentage_error(oar,par)


o_r = y_s.inverse_transform(test_y)
XGBrmse =rmse(o_r,XGBp_r)
XGBrmae =mae(o_r,XGBp_r)
#RandomForest模型

rf=RandomForestRegressor()
rf_params = {'max_depth':np.linspace(2,10,5,dtype=int)
             ,'n_estimators':np.linspace(50,250,5,dtype=int) }
tStart_rfm = time.time()
rf_g = GridSearchCV(rf , rf_params, verbose=2,refit=True, n_jobs=1 ,cv=cv)
rf_g.fit(train_X, train_y)
rf_best = rf_g.best_estimator_
importancesrf=rf_best.feature_importances_
for i,v in enumerate(importancesrf):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importancesrf))], importancesrf)
plt.show()
rfp= rf_best.predict(test_X)
tEnd_rfm = time.time() 
#反標準化   
rfp = np.array(rfp).reshape(-1,1)
rfp_r = y_s.inverse_transform(rfp)
rfrmse =rmse(o_r,rfp_r)
rfrmae =mae(o_r,rfp_r)
    
from sklearn.ensemble import AdaBoostRegressor

adab = AdaBoostRegressor()
adab_params = {'learning_rate':np.linspace(0.1,1,5,dtype=float)
               ,'n_estimators':np.linspace(100,500,5,dtype=int) }
tStart_ada= time.time()
adab = GridSearchCV(adab,adab_params,verbose=2,refit=True, n_jobs=1 ,cv=cv)
adab.fit(train_X, train_y)
ada_best = adab.best_estimator_
importancesada=ada_best.feature_importances_
for i,v in enumerate(importancesada):
	print('Feature: %0d, Score: %.5f' % (i,v))
plt.bar([x for x in range(len(importancesada))], importancesada)
plt.show()
adabp=ada_best.predict(test_X)
tEnd_ada = time.time()  
adabp= np.array(adabp).reshape(-1,1)
adabp_r = y_s.inverse_transform(adabp)
adarmse=rmse(o_r,adabp_r)
adamae=mae(o_r,adabp_r)


print(f"XGB最佳準確率: {gs_m.best_score_}，最佳參數組合：{gs_m.best_params_}")  
print(f"RF最佳準確率: {rf_g.best_score_}，最佳參數組合：{rf_g.best_params_}")  
print(f"ADA最佳準確率: {adab.best_score_}，最佳參數組合：{adab.best_params_}")  
df_c = pd.DataFrame({  'XG_rmse':[XGBrmse],'XG_mae':[XGBrmae]
                     ,'RF_rmse':[rfrmse],'RF_mae':[rfrmae]
                     ,'ada_rmse':[adarmse],'ada_mae':[adamae]
                          }) 
print("Train：",df_c)
xgtimes = tEnd_xgm - tStart_xgm
rftimes = tEnd_rfm - tStart_rfm
adatimes = tEnd_ada - tStart_ada
df_t = pd.DataFrame({  'XG_time':[xgtimes],'RF_time':[rftimes]
                     ,'ada_time':[adatimes]
                          }) 
print("Train_time：",df_t)
#-------------測驗集
Xt = data_t.iloc[:,2:15]
data_te = pd.DataFrame(Xt)
data_te['Seasons']= labelencoder.fit_transform(data_te['Seasons'])
data_te['Holiday']= labelencoder.fit_transform(data_te['Holiday'])
data_te['Functioning Day']= labelencoder.fit_transform(data_te['Functioning Day'])
Xt = data_te
# Xt = Xt.drop("Holiday",axis=1)
# Xt = Xt.drop("Snowfall (cm)",axis=1)
Xt = Xt.drop("Visibility (10m)",axis=1) #重複
Xt = X_s.fit_transform(Xt)
result_xgb = gs_best.predict(Xt)
result_rf = rf_best.predict(Xt)
result_ada = ada_best.predict(Xt)
result_xgb= np.array(result_xgb).reshape(-1,1)
result_rf= np.array(result_rf).reshape(-1,1)
result_ada= np.array(result_ada).reshape(-1,1)
result_xgb = y_s.inverse_transform(result_xgb)
result_rf = y_s.inverse_transform(result_rf)
result_ada = y_s.inverse_transform(result_ada)
yt = data_t['Rented Bike Count']
xgbrmse_p=rmse(yt,result_xgb)
xgbmae_p=mae(yt,result_xgb)
rfrmse_p=rmse(yt,result_rf)
rfmae_p=mae(yt,result_rf)
adarmse_p=rmse(yt,result_ada)
adamae_p=mae(yt,result_ada)
df_pp = pd.DataFrame({  'XG_rmse':[xgbrmse_p],'XG_mae':[xgbmae_p]
                     ,'RF_rmse':[rfrmse_p],'RF_mae':[rfmae_p]
                     ,'ada_rmse':[adarmse_p],'ada_mae':[adamae_p]
                          }) 
print("Test：", df_pp)
result_xgb = pd.DataFrame(result_xgb)
result_rf = pd.DataFrame(result_rf)
result_ada = pd.DataFrame(result_ada)
result = pd.concat([result_xgb,result_rf,result_ada],axis=1)
result=pd.DataFrame(result)
result.to_csv('DM2Bike_ans.csv',index = False)
yt.to_csv('DM2Bike_ori.csv',index = False)
    
    
    
