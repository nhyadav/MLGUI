from django.urls import path
from .views import *


urlpatterns = [
    # path('', home, name="home"),
    path('get_description_data/',getDescriptionData, name="getDescriptionData"),
    path('get_plot_data/',getPlotData, name="getPlotData"),
    path('get_heatmap_data/',get_heatmap_data,name='get_heatmap_data'),
    path("feature_engineering/",get_feature_engineering, name="feature_engineering"),
    path("feature_selection/",get_feature_selection, name="feature_selection"),
    path("feature_sacaling/",get_feature_sacaling, name="feature_sacaling"),
    path("feature_reduction/",get_feature_reduction, name="feature_reduction"),
    path("feature_transformation/",get_feature_transformation,name='feature_transformation'),

    #data preprocessing
    path("data_preprocessing/", get_data_processing, name='data_preprocessing'),
    path('imputation_preprocessing/', get_imputation_preprocess, name='imputation_process'),
    path('transformation_preprocessing/',get_transformation_preprocessing, name='transformation_preprocessing'),
    path('data_splitting/', get_data_splitting, name='data_splitting'),
    path('traintestsplit/',get_traintestsplit, name='traintest'),

    ########
    path('linearregression/',build_linearregression, name='linearregression'),
    path('lassoregression/',build_lassoregression, name='lassoregression'),
    path('ridgeregression/',build_ridgeregression, name='ridgeregression'),
    path('dtregression/',build_dtregression, name='dtregression'),
    path('randomregression/',build_randomregression, name='randomregression'),
    path('svrregression/',build_svrregression, name='svrregression'),
    path('knnregression/',build_knnregression, name='knnregression'),
    path('xgbregression/',build_xgbregression, name='xgbregression'),


################classifier
    path('logisticregression/',build_logisticregression, name='logisticregression'),
    path('dtclassifier/',build_dtclassifier, name='dtclassifier'),
    path('randomclassifier/',build_randomclassifier, name='randomclassifier'),
    path('svcclassifier/',build_svcclassifier, name='svcclassifier'),
    path('knnclassifier/',build_knnclassifier, name='knnclassifier'),
    path('xgbclassifier/',build_xgbclassifier, name='xgbclassifier'),

##########Evalution
    path('evalution_metrics/',get_evalution_metrics,name='evalution_metrics')







]
