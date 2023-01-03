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
