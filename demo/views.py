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
# Create your views here.

logger = logging.getLogger(__name__)
base_dir  = Path(__file__).resolve().parent.parent
rawdatapath = os.path.join(base_dir,'config/rawdata.pkl')

def home(request):
    logger.info("This is First Logger setup-in Django.")
    logger.debug("this is debugging")
    return render(request, 'index.html')

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def getDescriptionData(request):
    try:
        user = request.user
        user_data = Dataset.objects.filter(user=user).first()
        if user_data:
            try:
                file_data = request.FILES.get('file')
                Dataset(user=user, file=file_data).save()
                path = 'media/datasets/'+str(file_data.name)
                dp = DataPreprocessing(path)
                dp.load_data()
                dp.getsummary()
                dp.get_stats_description()
                with open(rawdatapath, "wb") as fout:
                    pickle.dump(dp.data, fout)
                # print(dp.stats_description)
                response = [dp.data_,dp.datasummary_,dp.stats_description]
            except Exception as e:
                print("Anything is wrong",e)
                logger.error(e)
                response = [{'error':e}]
            return Response(response, status=200)
        else:
            file_data = request.FILES.get('file')
            usr_dt = Dataset.objects.create(user=user,file=file_data)
            serializer = DatasetSerializer(usr_dt, many=False)
        return Response(serializer.data)
    except Dataset.DoesNotExist:
        return None
    
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
        return Response([{'error':e}])


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
        return Response([{'error':e}])

@csrf_exempt
@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_feature_engineering(request):
    with open(rawdatapath, 'rb') as out:
        data = pickle.load(out)
    return Response({'data':data.fillna('').values,'features':data.columns})

    

@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_feature_sacaling(request):
    content = json.loads(request.body)
    operation = content['operation']['scaleAttribte']
    print(operation)
    with open(rawdatapath, 'rb') as out:
        data = pickle.load(out)
    if operation == "--Select--":
        return Response({'data':data.fillna('').values,'features':data.columns})

    elif operation == "min-max":
        mns = MinMaxScaler()
        try:
            sc_data = mns.fit_transform(data.select_dtypes(include=[np.int64, np.float64]))
            sc_data = pd.DataFrame(sc_data, columns=data.select_dtypes(include=[np.int64, np.float64]).columns)
            for col in data.select_dtypes(include=[np.int64, np.float64]).columns:
                data[col] = sc_data[col]
            print("done....")
            with open(rawdatapath,'wb') as out:
                pickle.dump(data, out)
            return Response({'data':data.fillna('').values,'features':data.columns})
        except Exception as e:
            return Response({'data':data.fillna('').values,'features':data.columns})
    return Response({'data': None})



@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_feature_reduction(request):
    content = json.loads(request.body)
    print("Content of Feature reduction: ", content)
    with open(rawdatapath, 'rb') as out:
        data = pickle.load(out)
    components = 2
    pca = PCA(
        n_components=components
    )
    pca.fit(data.select_dtypes(include=[np.int64, np.float64]))
    x_pca = pca.transform(data.select_dtypes(include=[np.int64, np.float64]))
    return Response({'status': "Success", "code":200, 'data':x_pca})


@csrf_exempt
@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_feature_transformation(request):
    try:
        content = json.loads(request.body)
        operation = content['operation']['transAttribute']
        print("Feature Transformation....", content)
        with open(rawdatapath,'rb') as out:
            data = pickle.load(out)
        if operation == 'std':
            std = StandardScaler()
            sc_data = std.fit_transform(data.select_dtypes(exclude=['object','O']))
            sc_data = pd.DataFrame(sc_data, columns=data.select_dtypes(exclude=['object','O']).columns)
            for col in data.select_dtypes(exclude=['object','O']).columns:
                data[col] = sc_data[col]
            with open(rawdatapath, 'wb') as out:
                pickle.dump(data, out)
            return Response({'data':data.fillna('').values,'features':data.columns})
        elif operation == 'box-cox':
            pass
        else:
            return Response({'Status':'Not a Valid Operation','Code':200,'data':None})
    except Exception as e:
        print("hesfnkds", e)
        return Response({'Status':'Error',"Code":200, 'Error': e})