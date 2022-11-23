from django.urls import path
from .views import *


urlpatterns = [
    path('', home, name="home"),
    path('get_description_data/',getDescriptionData, name="getDescriptionData"),
]
