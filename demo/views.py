from django.shortcuts import render,get_object_or_404
from django.http import JsonResponse
import logging
from django.views.decorators.csrf import csrf_exempt
from rest_framework.response import Response
from rest_framework.decorators import api_view, permission_classes
from rest_framework.permissions import IsAuthenticated
from .serializer import DatasetSerializer
from .models import Dataset
from rest_framework.parsers import JSONParser
import json
from src.data_description import DataPreprocessing
# Create your views here.

logger = logging.getLogger(__name__)


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
                response = [dp.data_,dp.datasummary_]
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
    