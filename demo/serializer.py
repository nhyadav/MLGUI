from rest_framework.serializers import ModelSerializer
from .models import Dataset


class DatasetSerializer(ModelSerializer):
    class Meta:
        model=Dataset
        fields = '__all__'