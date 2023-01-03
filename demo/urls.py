from django.urls import path
from .views import *


urlpatterns = [
    path('', home, name="home"),
    path('get_description_data/',getDescriptionData, name="getDescriptionData"),
    path('get_plot_data/',getPlotData, name="getPlotData"),
    path('get_heatmap_data/',get_heatmap_data,name='get_heatmap_data')
]
