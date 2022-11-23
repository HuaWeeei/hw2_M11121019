# -*- coding: utf-8 -*-
"""
Created on Sun Sep 18 12:39:32 2022

@author: bbill
"""
import pandas as pd
import numpy as np
from xgboost import XGBRFRegressor
import time
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


df = pd.read_csv("adult_train.csv")
dt = pd.read_csv("adult_train.csv")
X1 = df.iloc[:,0:12]
X2 = df.iloc[:,13:15]
X= pd.concat([X1,X2],axis=1)
y = df['hours-per-week']


X = X.drop("capital-gain",axis=1)
X = X.drop("capital-loss",axis=1)
# X = X.drop("sex",axis=1) #8
# X = X.drop("native-country",axis=1) #12
labelencoder = LabelEncoder() #類別化
data_le = pd.DataFrame(X)
data_le['workclass']= labelencoder.fit_transform(data_le['workclass'])
data_le['education']= labelencoder.fit_transform(data_le['education'])
data_le['marital-status']= labelencoder.fit_transform(data_le['marital-status'])
data_le['occupation']= labelencoder.fit_transform(data_le['occupation'])
data_le['relationship']= labelencoder.fit_transform(data_le['relationship'])
data_le['sex']= labelencoder.fit_transform(data_le['sex'])
data_le['native-country']= labelencoder.fit_transform(data_le['native-country'])
data_le['race']= labelencoder.fit_transform(data_le['race'])
data_le['target']= labelencoder.fit_transform(data_le['target'])
X = data_le

X_s = StandardScaler()       
X = X_s.fit_transform(X)
y = pd.DataFrame(y)
y_s = StandardScaler()
y = y_s.fit_transform(y)
train_X, test_X, train_y, test_y = train_test_split( X, y,test_size=0.2,random_state=(42))

cv=5

xgbrModel=XGBRFRegressor()
cv_params = {'max_depth':np.linspace(2,10,5,dtype=int)
            ,'n_estimators':np.linspace(500,1000,5,dtype=int),'eta':[0.1,0.05,0.01] }
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
from sklearn import metrics
def rmse(oar,par):
    return metrics.mean_squared_error(oar,par)**0.5
def mae(oar,par):
    return metrics.mean_absolute_error(oar,par)
def mape(oar,par):
    return metrics.mean_absolute_percentage_error(oar,par)
o_r = y_s.inverse_transform(test_y)
xgtimes = tEnd_xgm - tStart_xgm
rftimes = tEnd_rfm - tStart_rfm
adatimes = tEnd_ada - tStart_ada
XGBrmse =rmse(o_r,XGBp_r)
XGBrmae =mae(o_r,XGBp_r)
XGBmape =mape(o_r,XGBp_r)
rfrmse =rmse(o_r,rfp_r)
rfmae =mae(o_r,rfp_r)
rfmape =mape(o_r,rfp_r)
adarmse=rmse(o_r,adabp_r)
adamae=mae(o_r,adabp_r)
adamape=mape(o_r,adabp_r)
print(f"XGB最佳準確率: {gs_m.best_score_}，最佳參數組合：{gs_m.best_params_}")  
print(f"RF最佳準確率: {rf_g.best_score_}，最佳參數組合：{rf_g.best_params_}")  
print(f"ADA最佳準確率: {adab.best_score_}，最佳參數組合：{adab.best_params_}")  
df_c = pd.DataFrame({  'XG_rmse':[XGBrmse],'XG_mae':[XGBrmae],'XG_mape':[XGBmape]
                     ,'RF_rmse':[rfrmse],'RF_mae':[rfmae],'RF_mape':[rfmape]
                     ,'ada_rmse':[adarmse],'ada_mae':[adamae],'ada_mape':[adamape]
                          }) 
print("Train：",df_c)
xgtimes = tEnd_xgm - tStart_xgm
rftimes = tEnd_rfm - tStart_rfm
adatimes = tEnd_ada - tStart_ada
df_t = pd.DataFrame({  'XG_time':[xgtimes],'RF_time':[rftimes]
                     ,'ada_time':[adatimes]
                          }) 
print("Train_time：",df_t)
#----------------------------------------------------測試集
Xt_1 = dt.iloc[:,0:12]
Xt_2 = dt.iloc[:,13:15]
Xt= pd.concat([Xt_1,Xt_2],axis=1)
data_dt = pd.DataFrame(Xt)
data_dt['workclass']= labelencoder.fit_transform(data_dt['workclass'])
data_dt['education']= labelencoder.fit_transform(data_dt['education'])
data_dt['marital-status']= labelencoder.fit_transform(data_dt['marital-status'])
data_dt['occupation']= labelencoder.fit_transform(data_dt['occupation'])
data_dt['relationship']= labelencoder.fit_transform(data_dt['relationship'])
data_dt['sex']= labelencoder.fit_transform(data_dt['sex'])
data_dt['native-country']= labelencoder.fit_transform(data_dt['native-country'])
data_dt['race']= labelencoder.fit_transform(data_dt['race'])
data_dt['target']= labelencoder.fit_transform(data_dt['target'])
Xt = data_dt
Xt = Xt.drop(["capital-gain","capital-loss"
              # ,"sex","native-country"
              ],axis=1)
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
yt = dt['hours-per-week']
xgbrmse_p=rmse(yt,result_xgb)
xgbmae_p=mae(yt,result_xgb)
xgbmape_p =mape(o_r,XGBp_r)
rfrmse_p=rmse(yt,result_rf)
rfmae_p=mae(yt,result_rf)
rfmape_p =mape(o_r,rfp_r)
adarmse_p=rmse(yt,result_ada)
adamae_p=mae(yt,result_ada)
adamape_p=mape(yt,result_ada)
df_pp = pd.DataFrame({  'XG_rmse':[xgbrmse_p],'XG_mae':[xgbmae_p],'XG_mape':[xgbmape_p]
                     ,'RF_rmse':[rfrmse_p],'RF_mae':[rfmae_p],'RF_mape':[rfmape_p]
                     ,'ada_rmse':[adarmse_p],'ada_mae':[adamae_p],'ada_mape':[adamape]
                          }) 
print("Test：", df_pp)
result_xgb = pd.DataFrame(result_xgb)
result_rf = pd.DataFrame(result_rf)
result_ada = pd.DataFrame(result_ada)
result = pd.concat([result_xgb,result_rf,result_ada],axis=1)
result=pd.DataFrame(result)
result.to_csv('DM2Adult_ans.csv',index = False)
yt.to_csv('DM2Adult_ori.csv',index = False)
    

   
   