from django.shortcuts import render
# from django.http import JsonResponse
import logging
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from .serializer import DatasetSerializer
from .models import Dataset
# from rest_framework.parsers import JSONParser
import json
from src.data_description import DataPreprocessing
import pickle
from pathlib import Path,os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.decomposition import PCA
import yaml
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score,mean_absolute_error,mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor 
#######classification
from sklearn.metrics import accuracy_score,precision_score,recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
# Create your views here.


base_dir  = Path(__file__).resolve().parent.parent
rawdatapath = os.path.join(base_dir,'config/rawdata.pkl')
dpdatapath = os.path.join(base_dir,'config/dpdata.pkl')
# precdatapath = os.path.join(base_dir,'config/precddata.pkl')
data_proc = os.path.join(base_dir,'config/data')
configpath = os.path.join(base_dir,'config/config.yaml')

class_eval = os.path.join(base_dir,'config/matrics/modelEvalsClass.pkl')
regressio_eval = os.path.join(base_dir,'config/matrics/modelsEvals.pkl')

#######yaml
global params
with open(configpath, encoding='utf-8') as prms:
    params = yaml.safe_load(prms)


def save_config(p):
    with open(configpath, 'w', encoding='utf-8') as out:
        yaml.safe_dump(p,out)
    # with open(configpath, encoding='utf-8') as out:
    #     params = yaml.safe_load(out)









###############################
# /def home(request):
#     logger.info("This is First Logger setup-in Django.")
#     logger.debug("this is debugging")
#     return render(request, 'index.html')

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def getDescriptionData(request):
    user = request.user
    user_data = Dataset.objects.filter(user=user).first()
    with open(configpath, encoding='utf-8') as out:
        params = yaml.safe_load(out)
    if user_data:
        try:
            file_data = request.FILES.get('file')
            Dataset(user=user, file=file_data).save()
            try:
                if os.environ['production_path']:
                    path = '~/MLGUI/media/datasets/'+str(file_data.name)
            except Exception as e:
                path = 'media/datasets/'+str(file_data.name)
            dp = DataPreprocessing(path)
            dp.load_data()
            dp.getsummary()
            dp.get_stats_description()
            with open(rawdatapath, "wb") as fout:
                pickle.dump(dp.data, fout)
            params['DataPreprocessing']['Imputation'] = False
            params['DataPreprocessing']['Transformation'] = False
            params['FeatureEngineering']['FeatureSelection'] = False
            params['FeatureEngineering']['FeatureScaling'] = False
            params['FeatureEngineering']['FeatureTransformation'] = False
            params['ModelEvalution']['Classification'] = False
            params['ModelEvalution']['Regression'] = False
            params['ModelEvalution']['Cluster'] = False
            params['ModelEvalution']['TSA'] = False
            save_config(params)
            response = [dp.data_,dp.datasummary_,dp.stats_description]
        except Exception as e:
            print("Anything is wrong",e)
            response = [{'error':e}]
        return Response(response, status=200)
    else:
        file_data = request.FILES.get('file')
        usr_dt = Dataset.objects.create(user=user,file=file_data)
        path = 'media/datasets/'+str(file_data.name)
        dp = DataPreprocessing(path)
        dp.load_data()
        dp.getsummary()
        dp.get_stats_description()
        with open(rawdatapath, "wb") as fout:
            pickle.dump(dp.data, fout)
        response = [dp.data_,dp.datasummary_,dp.stats_description]
        return Response(response, status=200)
    
    
@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def getPlotData(request):
    try:
        with open(rawdatapath, 'rb') as out:
            data = pickle.load(out)
        response_data = data.to_json(orient='records')
        num_columns = data.select_dtypes(include=[np.int64, np.float64]).columns
        cat_columns = data.select_dtypes(exclude=[np.int64, np.float64]).columns
        # num_columns = data.columns
        attributes = []
        cat_attribute = []
        for col in num_columns:
            attributes.append({'value':col, 'label':col})
        for col in cat_columns:
            cat_attribute.append({'value':col,'label':col})
        data = json.loads(response_data)
        # print(data)
        # print(type(json.loads(response_data)))
        # print(attributes)
        return Response({'data':data, 'attributes':attributes,'cat_attributes':cat_attribute})
    except Exception as e:
        logger.error('Error during fetching plot data:',e)
        return Response([{'error':str(e)}])


@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_heatmap_data(request):
    try:
        with open(rawdatapath, 'rb') as out:
            data = pickle.load(out)
        corr_features = data.corr().columns.to_list()
        corr_data = data.fillna('').corr().values
        null_data = data.isna().values
        return Response({'corr_features':corr_features,'corr_data':corr_data,'null_data':null_data})
    except Exception as e:
        logger.error('Error during fetching plot data:',e)
        return Response([{'error':str(e)}])


###Data preprocessing
@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_data_processing(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    try:
        oper = any([v for k,v in params['DataPreprocessing'].items()])
        # print('oper',oper)
        if not oper:
            with open(rawdatapath,'rb') as out:
                data = pickle.load(out)
        else:
            with open(dpdatapath,'rb') as out:
                data = pickle.load(out)
        return Response({'data':data.fillna('').values,'features':data.columns,'null_data':data.isnull().sum().to_dict()
        })
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Error':str(e)})
    
@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_imputation_preprocess(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    imputation = params['DataPreprocessing']['Imputation']
    if not imputation:
        oper = any([v for k,v in params['DataPreprocessing'].items()])
        if not oper:
            with open(rawdatapath,'rb') as out:
                data = pickle.load(out)
        else:
            with open(dpdatapath,'rb') as out:
                data = pickle.load(out)

        if data.isnull().values.any():
            print("imputation is called........")
            content = json.loads(request.body)
            operation = content['imputAttribute']
            print('operation',operation)
            if operation == 'drop':
                cols_missing = [col for col in data.columns if data[col].isnull().any()]
                print(cols_missing)
                if cols_missing:
                    data = data.drop(cols_missing, axis=1)
                    with open(dpdatapath, 'wb') as out:
                        pickle.dump(data, out)
            elif operation == 'droprows':
                data = data.dropna()
                with open(dpdatapath,'wb') as out:
                    pickle.dump(data, out)
            elif operation == 'mean':
                for col in data.select_dtypes(include=[np.int64,np.float64]).columns:
                    data[col] = data[col].fillna(data[col].mean())
                for col in data.select_dtypes(exclude=[np.int64,np.float64]).columns:
                    data[col] = data[col].fillna(data[col].mode()[0])
                with open(dpdatapath, 'wb') as out:
                    pickle.dump(data, out)
            elif operation == 'median':
                for col in data.select_dtypes(include=[np.int64,np.float64]).columns:
                    data[col] = data[col].fillna(data[col].median())
                for col in data.select_dtypes(exclude=[np.int64,np.float64]).columns:
                    data[col] = data[col].fillna(data[col].mode()[0])
                with open(dpdatapath, 'wb') as out:
                    pickle.dump(data, out)
            elif operation == 'mode':
                for col in data.select_dtypes(include=[np.int64,np.float64]).columns:
                    data[col] = data[col].fillna(data[col].mode()[0])
                for col in data.select_dtypes(exclude=[np.int64,np.float64]).columns:
                    data[col] = data[col].fillna(data[col].mode()[0])
                with open(dpdatapath, 'wb') as out:
                    pickle.dump(data, out)
            params['DataPreprocessing']['Imputation'] = True
            save_config(params)
        else:
            # params['DataPrerocessing']['Imputation'] = True
            return Response({'Status':'No Need','Msg':"No Need to Perform Imputation."})
        return Response({'Status':'Done'})
    else:
        return Response({'Status':'Already Done','Msg':'You Have Already perfomr imputation.'})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_transformation_preprocessing(request):
    try:
        with open(configpath, encoding='utf-8') as prms:
            params = yaml.safe_load(prms)
        features_transformation = params['DataPreprocessing']['Transformation']
        if not features_transformation:
            content = json.loads(request.body)
            print('content', content)
            oper = any([v for k,v in params['DataPreprocessing'].items()])
            if not oper:
                with open(rawdatapath,'rb') as out:
                    data = pickle.load(out)
            else:
                with open(dpdatapath,'rb') as out:
                    data = pickle.load(out)
            operation = content['transattribute']
            if operation == 'label encoder':
                lblencoder = LabelEncoder()
                for col in data.select_dtypes(exclude=[np.int64, np.float64]).columns:
                    data[col] = lblencoder.fit_transform(data[col])
                with open(dpdatapath, 'wb') as out:
                    pickle.dump(data, out)
                params['DataPreprocessing']['Transformation'] = True
                save_config(params)
            elif operation == 'categorical encoding':
                pass
            elif operation == 'one hot encoding':
                pass
            elif operation == 'get dummies':
                data = pd.get_dummies(data)
                with open(dpdatapath,'wb') as out:
                    pickle.dump(data, out)
                params['DataPreprocessing']['Transformation'] = True
                save_config(params)

            return Response({'Status':'Done','Msg':'Transformation Done....'})

        else:
            return Response({'Status':'Already Done','Msg':"You Have Already Performed Transformar"})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Error':str(e)})








########################
############Feature Engineering###### 
@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_feature_engineering(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    try:
        oper = any([v for k,v in params['DataPreprocessing'].items()])
        foper = any([v for k,v in params['FeatureEngineering'].items()])
        # print('oper',oper)
        if not (oper or foper):
            with open(rawdatapath,'rb') as out:
                data = pickle.load(out)
        else:
            with open(dpdatapath,'rb') as out:
                data = pickle.load(out)
        return Response({'data':data.fillna('').values,'features':data.columns})
    except Exception as e:
        return Response({'Status':'Failed','Msg':str(e)})

    
@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_feature_selection(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    features_selection = params['FeatureEngineering']['FeatureSelection']
    try:
        if not features_selection:
            content = json.loads(request.body)
            columns = content['selectcols']['columns']
            print("Feature selection:",columns)
            oper = any([v for k,v in params['DataPreprocessing'].items()])
            foper = any([v for k,v in params['FeatureEngineering'].items()])
            if not (oper or foper):
                with open(rawdatapath,'rb') as out:
                    data = pickle.load(out)
            else:
                with open(dpdatapath,'rb') as out:
                    data = pickle.load(out)
            # print(data.head())
            with open(dpdatapath, 'wb') as out:
                pickle.dump(data.loc[:,columns],out)
            params['FeatureEngineering']['FeatureSelection'] = True
            save_config(params)
            return Response({'Status':"Done",'Msg':"Feature selection have Done!!!"})
        else:
            return Response({'Status':'Already Done','Msg':'You Have Already Performed Features Selection.'})
    except Exception as e:
        print('error',e)
        return Response({"Status":'Failed','Error':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_feature_sacaling(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    scaling_done = params['FeatureEngineering']['FeatureScaling']
    try:
        if not scaling_done:
            content = json.loads(request.body)
            operation = content['scaleAttribute']
            oper = any([v for k,v in params['DataPreprocessing'].items()])
            foper = any([v for k,v in params['FeatureEngineering'].items()])
            if not (oper or foper):
                with open(rawdatapath,'rb') as out:
                    data = pickle.load(out)
            else:
                with open(dpdatapath,'rb') as out:
                    data = pickle.load(out)
            # if operation == "--Select--":
            #     return Response({'data':data.fillna('').values,'features':data.columns})
            if operation == "min-max":
                mns = MinMaxScaler()
                sc_data = mns.fit_transform(data.select_dtypes(include=[np.int64, np.float64]))
                sc_data = pd.DataFrame(sc_data, columns=data.select_dtypes(include=[np.int64, np.float64]).columns)
                for col in data.select_dtypes(include=[np.int64, np.float64]).columns:
                    data[col] = sc_data[col]
                print("done....")
                with open(dpdatapath,'wb') as out:
                    pickle.dump(data, out)
                # global scaling_done
                params['FeatureEngineering']['FeatureScaling'] = True
                save_config(params)
                return Response({'data':data.fillna('').values,'features':data.columns})
        else:
            return Response({'Status':'Already Done','Msg':'You Have Already Performed Features Scaling.'})
    except Exception as e:
        print('sss',e)
        return Response({'data':data.fillna('').values,'features':data.columns, 'Status':'Failed','Error':str(e)})



@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_feature_reduction(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    try:
        content = json.loads(request.body)
        print("Content of Feature reduction: ", content)
        oper = any([v for k,v in params['DataPreprocessing'].items()])
        foper = any([v for k,v in params['FeatureEngineering'].items()])
        # print('oper',oper)
        if not (oper or foper):
            with open(rawdatapath,'rb') as out:
                data = pickle.load(out)
        else:
            with open(dpdatapath,'rb') as out:
                data = pickle.load(out)
        components = 2
        pca = PCA(
            n_components=components
        )
        pca.fit(data.select_dtypes(include=[np.int64, np.float64]))
        x_pca = pca.transform(data.select_dtypes(include=[np.int64, np.float64]))
        return Response({'Status': "Success", "code":200, 'data':x_pca})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Error':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_feature_transformation(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    transformation_fe = params['FeatureEngineering']['FeatureTransformation']
    if not transformation_fe:
        try:
            content = json.loads(request.body)
            operation = content['operation']['transAttribute']
            print("Feature Transformation....", content)
            oper = any([v for k,v in params['DataPreprocessing'].items()])
            foper = any([v for k,v in params['FeatureEngineering'].items()])
            # print('oper',oper)
            if not (oper or foper):
                with open(rawdatapath,'rb') as out:
                    data = pickle.load(out)
            else:
                with open(dpdatapath,'rb') as out:
                    data = pickle.load(out)
            if operation == 'std':
                std = StandardScaler()
                sc_data = std.fit_transform(data.select_dtypes(exclude=['object','O']))
                sc_data = pd.DataFrame(sc_data, columns=data.select_dtypes(exclude=['object','O']).columns)
                for col in data.select_dtypes(exclude=['object','O']).columns:
                    data[col] = sc_data[col]
                with open(dpdatapath, 'wb') as out:
                    pickle.dump(data, out)
                params['FeatureEngineering']['FeatureTransformation'] = True
                save_config(params)
                return Response({'data':data.fillna('').values,'features':data.columns})
            elif operation == 'box-cox':
                for col in data.select_dtypes(include=[np.int64,np.float64]).columns:
                    data[col] = stats.boxcox(data[col])[0]
                with open(dpdatapath,'wb') as out:
                    pickle.dump(data, out)
                params['FeatureEngineering']['FeatureTransformation'] = True
                save_config(params)
                return Response({'data':data.fillna('').value,'features':data.columns})
            else:
                return Response({'Status':'Not a Valid Operation','Code':200,'data':None})
        except Exception as e:
            print("hesfnkds", e)
            return Response({'Status':'Failed', 'Error': e})
    else:
        return Response({'Status':'Already Done','Msg':'You Have Already Performed Features Transformer.'})

###########################
##########data spliting and model selection#########3
@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_data_splitting(request):
    try:
        with open(dpdatapath, 'rb') as out:
            data = pickle.load(out)
        features = []
        for col in data.columns:
            features.append({'value':col,'label':col})    
        return Response({"Status":'Done','Features':features})
    except Exception as e:
        return Response({'Status':'Failed','Msg':str(e)})
    

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_traintestsplit(request):
    try:
        content = json.loads(request.body)
        targetcols = content['targetcol']
        testcols = content['testprob']
        with open(dpdatapath,'rb') as out:
            data = pickle.load(out)
        x_train,x_test,y_train,y_test = train_test_split(data.drop([targetcols],axis=1),data.loc[:,targetcols],test_size=float(testcols),random_state=42)
        with open(data_proc+'/x_train.pkl','wb') as out:
            pickle.dump(x_train, out)    
        with open(data_proc+'/x_test.pkl','wb') as out:
            pickle.dump(x_test, out)    
        with open(data_proc+'/y_train.pkl','wb') as out:
            pickle.dump(y_train, out)
        with open(data_proc+'/y_test.pkl','wb') as out:
            pickle.dump(y_test, out)
        # print('content',content)
        print('shape x_train', x_train.shape)        
        print('shape y_train',y_train.shape)
        print('shape x_test',x_test.shape)        
        print('shape y_test',y_test.shape)
        return Response({'Status':'Done','Msg':'Successfully done Train Test Split....'})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Error':str(e)})
    




@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_linearregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        savemodel = content['save']
        print("content",content)
        if savemodel:
            lrm = LinearRegression()
            with open(data_proc+'/x_train.pkl','rb') as out:
                x_train = pickle.load(out)        
            with open(data_proc+'/x_test.pkl','rb') as out:
                x_test = pickle.load(out)
            with open(data_proc+'/y_train.pkl','rb') as out:
                y_train = pickle.load(out)
            with open(data_proc+'/y_test.pkl','rb') as out:
                y_test = pickle.load(out)
            lrm.fit(x_train, y_train)
            pred = lrm.predict(x_test)
            rmse = mean_squared_error(y_test,pred,squared=False)
            mse = mean_squared_error(y_test,pred)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test,pred)
            adj_r2 = 0
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Linear Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['Linear Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)

            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'Linear Regression Model is Trained.'})
        else:
            lrm = LinearRegression()
            with open(data_proc+'/x_train.pkl','rb') as out:
                x_train = pickle.load(out)
            with open(data_proc+'/x_test.pkl','rb') as out:
                x_test = pickle.load(out)
            with open(data_proc+'/y_train.pkl','rb') as out:
                y_train = pickle.load(out)
            with open(data_proc+'/y_test.pkl','rb') as out:
                y_test = pickle.load(out)
            lrm.fit(x_train, y_train)
            pred = lrm.predict(x_test)
            rmse = mean_squared_error(y_test,pred,squared=False)
            mse = mean_squared_error(y_test,pred)
            mae = mean_absolute_error(y_test, pred)
            r2 = r2_score(y_test,pred)
            adj_r2 = 0
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'Linear Regression Model is Trained.'})
    except Exception as e:
        return Response({'Status':'Failed','Msg':str(e)})
    


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_lassoregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = Lasso()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        rmse = mean_squared_error(y_test,pred,squared=False)
        mse = mean_squared_error(y_test,pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)
        adj_r2 = 0
        if savemodel:
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Lasso Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['Lasso Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'Lasso Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'Lasso Regression Model is Trained.'})
    except Exception as e:
        return Response({'Status':'Failed','Msg':str(e)})

@csrf_exempt
@api_view(["POST"])   
@permission_classes([IsAuthenticated])
def build_ridgeregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = Ridge()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        rmse = mean_squared_error(y_test,pred,squared=False)
        mse = mean_squared_error(y_test,pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)
        adj_r2 = 0
        if savemodel:
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Ridge Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['Ridge Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'Ridge Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'Ridge Regression Model is Trained.'})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Msg':str(e)})

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])    
def build_dtregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = DecisionTreeRegressor()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        rmse = mean_squared_error(y_test,pred,squared=False)
        mse = mean_squared_error(y_test,pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)
        adj_r2 = 0
        if savemodel:
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Decision Tree Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['Decision Tree Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'Decision Tree Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'Decision Tree Regression Model is Trained.'})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Msg':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_randomregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = RandomForestRegressor()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        rmse = mean_squared_error(y_test,pred,squared=False)
        mse = mean_squared_error(y_test,pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)
        adj_r2 = 0
        if savemodel:
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Random Forest Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['Random Forest Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'Random Forest Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'Random Forest Regression Model is Trained.'})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Msg':str(e)})




@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_svrregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = SVR()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        rmse = mean_squared_error(y_test,pred,squared=False)
        mse = mean_squared_error(y_test,pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)
        adj_r2 = 0
        if savemodel:
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['SVM Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['SVM Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'SVM Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'SVM Regression Model is Trained.'})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Msg':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_knnregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = KNeighborsRegressor()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        rmse = mean_squared_error(y_test,pred,squared=False)
        mse = mean_squared_error(y_test,pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)
        adj_r2 = 0
        if savemodel:
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['KNN Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['KNN Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'KNN Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'KNN Regression Model is Trained.'})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Msg':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_xgbregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    regression_m = params['ModelEvalution']['Regression']
    try:
        columns = ['Model Name','RMSE','MAE','MSE','R2','Adjested-R2']
        content = json.loads(request.body)
        print('content',content)
        savemodel = content['save']
        lrm = XGBRegressor()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        rmse = mean_squared_error(y_test,pred,squared=False)
        mse = mean_squared_error(y_test,pred)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test,pred)
        adj_r2 = 0
        if savemodel:
            if regression_m:
                print('ram')
                with open(regressio_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['XGB Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)])
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('ramji')
                f_df_e = pd.DataFrame([['XGB Regression',rmse,mse,mae,r2,adj_r2]], columns=columns)
                with open(regressio_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Regression'] = True
                save_config(params)
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,'Save':'Saved',"Msg":'XGB Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','RMSE':rmse,'MSE':mse,'MAE':mae,'R2':r2,'AdjR2':adj_r2,"Msg":'XGB Regression Model is Trained.'})
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Msg':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_logisticregression(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    classify_e = params['ModelEvalution']['Classification']
    try:
        columns = ['Model Name','Accuracy','Precision','Recall','F1_Score']
        content = json.loads(request.body)
        savemodel = content['save']
        print(content)
        lrm = LogisticRegression()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred,average='micro')
        recall = recall_score(y_test, pred,average='micro')
        f1_score = f1_score(y_test,pred,average='micro')
        if savemodel:
            if classify_e:
                with open(class_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Logistic Regression',accuracy,precision,recall,f1_score]], columns=columns)])
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                f_df_e = pd.DataFrame([['Logistic Regression',accuracy,precision,recall,f1_score]], columns=columns)
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Classification'] = True
                save_config(params)
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,'Save':'Saved',"Msg":'Logistic Regression Model is Trained.'})
        else:
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,"Msg":'logistic Regression Model is Trained.'})
    except Exception as e:
        print('error',e)
        return Response({'Status':'Failed','Error':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_dtclassifier(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    classify_e = params['ModelEvalution']['Classification']

    try:
        columns = ['Model Name','Accuracy','Precision','Recall','F1_Score']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = DecisionTreeClassifier()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred,average='micro')
        recall = recall_score(y_test, pred,average='micro')
        f1_score = f1_score(y_test,pred,average='micro')
        if savemodel:
            if classify_e:
                with open(class_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Decision Tree Classifier',accuracy,precision,recall,f1_score]], columns=columns)])
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                f_df_e = pd.DataFrame([['Decision Tree Classifier',accuracy,precision,recall,f1_score]], columns=columns)
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Classification'] = True
                save_config(params)
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,'Save':'Saved',"Msg":'Decision Tree Classifier Model is Trained.'})
        else:
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,"Msg":'Decision Tree Classifier Model is Trained.'})
    except Exception as e:
        print('error',e)
        return Response({'Status':'Failed','Error':str(e)})



@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_randomclassifier(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    classify_e = params['ModelEvalution']['Classification']

    try:
        columns = ['Model Name','Accuracy','Precision','Recall','F1_Score']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = RandomForestClassifier()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred,average='micro')
        recall = recall_score(y_test, pred,average='micro')
        f1_score = f1_score(y_test,pred,average='micro')
        if savemodel:
            if classify_e:
                with open(class_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Random Forest Classifier',accuracy,precision,recall,f1_score]], columns=columns)])
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                f_df_e = pd.DataFrame([['Random Forest Classifier',accuracy,precision,recall,f1_score]], columns=columns)
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Classification'] = True
                save_config(params)
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,'Save':'Saved',"Msg":'Random Forest Classifier Model is Trained.'})
        else:
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,"Msg":'Random Forest Classifier Model is Trained.'})
    except Exception as e:
        print('error',e)
        return Response({'Status':'Failed','Error':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_svcclassifier(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    classify_e = params['ModelEvalution']['Classification']

    try:
        columns = ['Model Name','Accuracy','Precision','Recall','F1_Score']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = SVC()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred,average='micro')
        recall = recall_score(y_test, pred,average='micro')
        f1_score = f1_score(y_test,pred,average='micro')
        if savemodel:
            if classify_e:
                with open(class_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Support Vectoer Classifier',accuracy,precision,recall,f1_score]], columns=columns)])
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                f_df_e = pd.DataFrame([['Support Vectoer Classifier',accuracy,precision,recall,f1_score]], columns=columns)
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Classification'] = True
                save_config(params)
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,'Save':'Saved',"Msg":'Support Vectoer Classifier Model is Trained.'})
        else:
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,"Msg":'Support Vectoer Classifier Model is Trained.'})
    except Exception as e:
        print('error',e)
        return Response({'Status':'Failed','Error':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_knnclassifier(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    classify_e = params['ModelEvalution']['Classification']
    try:
        columns = ['Model Name','Accuracy','Precision','Recall','F1_Score']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = KNeighborsClassifier()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred,average='micro')
        recall = recall_score(y_test, pred,average='micro')
        f1_score = f1_score(y_test,pred,average='micro')
        if savemodel:
            if classify_e:
                print('Jay shree Ram')
                with open(class_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['KNearest Neigbour',accuracy,precision,recall,f1_score]], columns=columns)])
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                print('Jay shree Ram34')
                f_df_e = pd.DataFrame([['KNearest Neigbour',accuracy,precision,recall,f1_score]], columns=columns)
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Classification'] = True
                save_config(params)
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,'Save':'Saved',"Msg":'KNearest Neigbour Model is Trained.'})
        else:
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,"Msg":'KNearest Neigbour Model is Trained.'})
    except Exception as e:
        print('error',e)
        return Response({'Status':'Failed','Error':str(e)})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def build_xgbclassifier(request):
    with open(configpath, encoding='utf-8') as prms:
        params = yaml.safe_load(prms)
    classify_e = params['ModelEvalution']['Classification']
    try:
        columns = ['Model Name','Accuracy','Precision','Recall','F1_Score']
        content = json.loads(request.body)
        savemodel = content['save']
        lrm = XGBClassifier()
        with open(data_proc+'/x_train.pkl','rb') as out:
            x_train = pickle.load(out)        
        with open(data_proc+'/x_test.pkl','rb') as out:
            x_test = pickle.load(out)
        with open(data_proc+'/y_train.pkl','rb') as out:
            y_train = pickle.load(out)
        with open(data_proc+'/y_test.pkl','rb') as out:
            y_test = pickle.load(out)
        
        lrm.fit(x_train, y_train)
        pred = lrm.predict(x_test)
        accuracy = accuracy_score(y_test,pred)
        precision = precision_score(y_test,pred,average='micro')
        recall = recall_score(y_test, pred,average='micro')
        f1_score = f1_score(y_test,pred,average='micro')
        if savemodel:
            if classify_e:
                with open(class_eval,'rb') as out:
                    df_e = pickle.load(out)
                f_df_e = pd.concat([df_e, pd.DataFrame([['Xgboost Classifier',accuracy,precision,recall,f1_score]], columns=columns)])
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
            else:
                f_df_e = pd.DataFrame([['Xgboost Classifier',accuracy,precision,recall,f1_score]], columns=columns)
                with open(class_eval,'wb') as out:
                    pickle.dump(f_df_e,out)
                params['ModelEvalution']['Classification'] = True
                save_config(params)
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,'Save':'Saved',"Msg":'Xgboost Classifier Model is Trained.'})
        else:
            return Response({'Status':'Done','Accuracy':accuracy,'Precision':precision,'Recall':recall,'F1_Score':f1_score,"Msg":'Xgboost Classifier Model is Trained.'})
    except Exception as e:
        print('error',e)
        return Response({'Status':'Failed','Error':str(e)})


@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_evalution_metrics(request):
    try:
        with open(configpath, encoding='utf-8') as prms:
            params = yaml.safe_load(prms)
        classification = params['ModelEvalution']['Classification']
        regression = params['ModelEvalution']['Regression']
        if classification or regression:
            if classification and regression:
                with open(class_eval, 'rb') as out:
                    class_df = pickle.load(out)
                with open(regressio_eval,'rb') as out:
                    regress_df = pickle.load(out)
                return Response({'Status':'Done','class_mv':class_df.values,'class_m':class_df.columns,'regre_mv':regress_df.values, 'regre_m':regress_df.columns})
            elif classification:
                with open(class_eval, 'rb') as out:
                    class_df = pickle.load(out)
                return Response({'Status':'Done','class_mv':class_df.values,'class_m':class_df.columns,'regre_mv':None, 'regre_m':None})
            elif regression:
                with open(regressio_eval,'rb') as out:
                    regress_df = pickle.load(out)
                return Response({'Status':'Done','class_mv':None,'class_m':None,'regre_mv':regress_df.values, 'regre_m':regress_df.columns})
        else:
            return Response({'status':'Done','class_m':None,'regre_m':None})   
    except Exception as e:
        print(e)
        return Response({'Status':'Failed','Msg':str(e)})










