from django.urls import path
from .views import *


urlpatterns = [
    path('', home, name="home"),
    path('get_description_data/',getDescriptionData, name="getDescriptionData"),
    path('get_plot_data/',getPlotData, name="getPlotData"),
    path('get_heatmap_data/',get_heatmap_data,name='get_heatmap_data'),
    path("feature_engineering/",get_feature_engineering, name="feature_engineering"),
    path("feature_sacaling/",get_feature_sacaling, name="feature_sacaling"),

    path("feature_reduction/",get_feature_reduction, name="feature_reduction"),
    path("feature_transformation/",get_feature_transformation,name='feature_transformation'),

]
