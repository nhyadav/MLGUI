import pandas as pd
import numpy as np



class DataPreprocessing:
    def __init__(self, path) -> None:
        self.path = path
    
    
    def load_data(self):
        self.data_={}
        if self.path.endswith(".csv"):
            self.data = pd.read_csv(self.path)
            self.data_['data']=self.data.values
            self.data_['features_name']=self.data.columns
        else:
            self.data = pd.read_excel(self.path)
            self.data_['data']=self.data.values
            self.data_['features_name']=self.data.columns
    
    
    def getsummary(self):
        self.datasummary_ = {}
        self.datasummary_['No.Rows']=self.data.shape[0]
        self.datasummary_['No.Cols']=self.data.shape[1]
        self.datasummary_['No.IntCols']=self.data.select_dtypes(include=[np.int64, np.float64]).shape[1]
        self.datasummary_['No.CatCols']=self.data.select_dtypes(exclude=[np.int64, np.float64]).shape[1]
        self.datasummary_['No.DateCols']=self.data.select_dtypes(exclude=[np.int64, np.float64, 'O','object']).shape[1]

        self.datasummary_['TotNull']=np.sum(self.data.isnull().values)
        self.datasummary_['TotpercNull']=round((np.sum(self.data.isnull().values)*100)/(self.data.shape[0]*self.data.shape[1]),2)
        self.datasummary_['TotCells']=self.data.shape[0]*self.data.shape[1]

    def get_stats_description(self):
        self.stats_description={}
        self.stats_description['features_name']=self.data.describe(include='all').reset_index().columns
        self.stats_description['stats_data']=self.data.describe(include='all').reset_index().values



