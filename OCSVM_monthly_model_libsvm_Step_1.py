import pandas as pd 
import numpy as np 
from libsvm.svmutil import *

quantile_value=0.75

data=pd.read_csv('mosquito_climate_pop_land_use_n.csv',low_memory=False)
start_index_t2m=data.columns.get_loc('t2m_1')
start_index_d2m=data.columns.get_loc('d2m_1')
start_index_tp=data.columns.get_loc('tp_1')
start_index_si10=data.columns.get_loc('si10_1')

t2m_mean=data.iloc[:,start_index_t2m:start_index_t2m+12].quantile(quantile_value,axis=1)
d2m_mean=data.iloc[:,start_index_d2m:start_index_d2m+12].quantile(quantile_value,axis=1)
tp_mean=data.iloc[:,start_index_tp:start_index_tp+12].quantile(quantile_value,axis=1)
si10_mean=data.iloc[:,start_index_si10:start_index_si10+12].quantile(quantile_value,axis=1)


data.insert(1,'t2m_mean',t2m_mean)
data.insert(2,'d2m_mean',d2m_mean)
data.insert(3,'tp_mean',tp_mean)
data.insert(4,'si10_mean',si10_mean)
data["Target_Class"]=1
df=data[data["VECTOR"]=="Aedes albopictus"]
#df_aegypti=df.drop(['Unnamed: 0','VECTOR','OCCURRENCE_ID','SOURCE_TYPE','LOCATION_TYPE','POLYGON_ADMIN','Y','X','COUNTRY','COUNTRY_ID','GAUL_AD0','STATUS','Population_Density','land_use_0','land_use_11','land_use_22','land_use_33','land_use_44','land_use_55','land_use_66','land_use_77'],axis=1)
#df_aegypti=df.drop(['Unnamed: 0','VECTOR','OCCURRENCE_ID','SOURCE_TYPE','LOCATION_TYPE','POLYGON_ADMIN','YEAR','Y','X','COUNTRY','COUNTRY_ID','GAUL_AD0','STATUS'],axis=1)
#columns_list=['t2m_mean', 'd2m_mean', 'tp_mean', 'si10_mean','Y','X','Population_Density', 'land_use_0', 'land_use_11',
#    'land_use_22', 'land_use_33', 'land_use_44', 'land_use_55',
#    'land_use_66', 'land_use_77','Target_Class'] # can/add remove Y,X from here
columns_list=['t2m_mean', 'd2m_mean', 'tp_mean', 'si10_mean','Population_Density', 'land_use_0', 'land_use_11',
    'land_use_22', 'land_use_33', 'land_use_44', 'land_use_55',
    'land_use_66', 'land_use_77','Target_Class'] # can/add remove Y,X from here
df_aegypti=df.loc[:,columns_list]
df_aegypti=df_aegypti.dropna()
#df_aegypti=df_aegypti[df_aegypti['Population_Density']>0]
X=df_aegypti.drop('Target_Class',axis=1)
y=df_aegypti['Target_Class']
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=2)
cols=X_train.columns
from sklearn.preprocessing import StandardScaler
#from sklearn.preprocessing import PowerTransformer
scaler=StandardScaler()
#scaler = PowerTransformer(method='yeo-johnson', standardize=True)  # Default: Yeo-Johnson and standardize=True
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
X_train=pd.DataFrame(X_train,columns=[cols])
X_test=pd.DataFrame(X_test,columns=[cols])
y_train=y_train.values.tolist()
X_train=X_train.values.tolist()
prob=svm_problem(y_train,X_train)
param=svm_parameter('-s 2 -t 2 -g 0.03 -n 0.03 -b 1') # n determines # of support vectors
m = svm_train(prob,param)
svm_save_model('ocsvm_albopictus_median_parameters_quantile_75.model', m)

for month in range(1,13):
    test_data_2024=pd.read_csv('2024_climate_data_subset_pop_den_land_use_lat_lon.csv',low_memory=False)
    test_data_2024['Target_Class']=-1
    test_data_2024=test_data_2024.drop(['Unnamed: 0'],axis=1)
    test_data_2024_pop=test_data_2024.copy()
    #test_data_2024_pop=test_data_2024[test_data_2024['Population_Density']>0]
    test_data_2024_pop=test_data_2024_pop.dropna()
    lat_test_data=test_data_2024_pop['Y'].copy()
    lon_test_data=test_data_2024_pop['X'].copy()
    geometry_test_data=test_data_2024_pop['geometry'].copy()
    #X_test_apparent=test_data_2024_pop.drop(['Y','X','YEAR','Target_Class','geometry'],axis=1)
    X_test_apparent=test_data_2024_pop.drop(['YEAR','Target_Class','geometry'],axis=1)
    y_test_apparent=test_data_2024_pop['Target_Class']

    var_t2m='t2m_'+str(month)
    var_d2m='d2m_'+str(month)
    var_tp='tp_'+str(month)
    var_si10='si10_'+str(month)

    start_index_t2m_monthly=X_test_apparent.columns.get_loc(var_t2m)
    start_index_d2m_monthly=X_test_apparent.columns.get_loc(var_d2m)
    start_index_tp_monthly=X_test_apparent.columns.get_loc(var_tp)
    start_index_si10_monthly=X_test_apparent.columns.get_loc(var_si10)

    t2m_val_mean=X_test_apparent.iloc[:,start_index_t2m_monthly:start_index_t2m_monthly+1].mean(axis=1)
    d2m_val_mean=X_test_apparent.iloc[:,start_index_d2m_monthly:start_index_d2m_monthly+1].mean(axis=1)
    tp_val_mean=X_test_apparent.iloc[:,start_index_tp_monthly:start_index_tp_monthly+1].mean(axis=1)
    si10_val_mean=X_test_apparent.iloc[:,start_index_si10_monthly:start_index_si10_monthly+1].mean(axis=1)


    X_test_apparent.insert(0,'t2m_mean',t2m_val_mean)
    X_test_apparent.insert(1,'d2m_mean',d2m_val_mean)
    X_test_apparent.insert(2,'tp_mean',tp_val_mean)
    X_test_apparent.insert(3,'si10_mean',si10_val_mean)
    #val_columns_list=['t2m_mean', 'd2m_mean', 'tp_mean', 'si10_mean','Y','X','Population_Density', 'land_use_0', 'land_use_11',
    #    'land_use_22', 'land_use_33', 'land_use_44', 'land_use_55',
    #    'land_use_66', 'land_use_77']
    val_columns_list=['t2m_mean', 'd2m_mean', 'tp_mean', 'si10_mean','Population_Density', 'land_use_0', 'land_use_11',
        'land_use_22', 'land_use_33', 'land_use_44', 'land_use_55',
        'land_use_66', 'land_use_77']
    X_test_apparent=X_test_apparent.loc[:,val_columns_list]
    X_test_transform=scaler.transform(X_test_apparent)
    X_test_apparent_original=scaler.inverse_transform(X_test_transform)
    X_test_apparent_original=pd.DataFrame(X_test_apparent_original,columns=[cols])
    X_test_apparent_original['Y']=lat_test_data.values
    X_test_apparent_original['X']=lon_test_data.values
    X_test_apparent_original['geometry']=geometry_test_data.values
    #X_test_apparent_original['Target_Class_Prediction']=y_pred_apparent
    #y_test=y_test.values.tolist()
    X_test_transform=X_test_transform.tolist()
    p_label, p_acc, p_val = svm_predict([], X_test_transform, m, '-b 1')

    df_prob=pd.DataFrame({'Latitude':lat_test_data.values,'Longitude':lon_test_data.values})
    df_prob['geometry']=geometry_test_data.values
    df_prob_new=pd.DataFrame(p_val,columns=['Prob_1', 'Prob_2'])
    df_prob_new['P_Label']=p_label
    combined_df=pd.concat([df_prob,df_prob_new],axis=1)
    filename='2024_monthly_mean_'+str(month)+'_ocsvm_albopictus_predictions_quantile_'+str(quantile_value)+'.csv'
    combined_df.to_csv(filename)

