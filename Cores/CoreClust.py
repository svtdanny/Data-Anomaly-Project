from DBConnector import DataBase
import pandas as pd
from Table import Table
import numpy as np
import pandas as pd
from os.path import isfile
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

import pickle

class CoreClust:
    def __init__(self, subject, targets, time_cols, source_db, result_db):
        self.subject = subject
        self.targets = targets
        self.time_cols = time_cols

        self.source_db = DataBase(source_db)
        self.result_db = DataBase(result_db)

        self.scaler = StandardScaler()

        self.modelers = {}
        self.model_path = './models/clust/'
    
    
    def fit_core(self, first=False):
        data = self.source_db.read(self.subject+'_source').get_data()
        time_col = data['Время']
        data = data[self.targets]

        data_scaled = self.scaler.fit_transform(data)

        isolation_forest = IsolationForest(n_estimators=160, contamination=0.05, 
                                   max_features=1.0, bootstrap=True, random_state=42)
        isolation_forest.fit(data_scaled)
        self.modelers['IsolationForest'] = isolation_forest
        self.save_models()

        if first:
            isolation_outliers = isolation_forest.predict(data_scaled)
            isolation_outliers = np.array([1 if label == -1 else 0 for label in isolation_outliers])
            
            df_result = pd.DataFrame(np.vstack([time_col, isolation_outliers]).T,
                                        columns=['time', 'outliers'])
            
            self.result_db.write_from_df(Table('Analitics_'+'clust_'+ self.subject, df_result),
                                            method='replace')


    def load_models(self):
        if isfile(self.model_path+'model_'+self.subject+'_IsolationForest'+'.txt'):
            with open(self.model_path+'model_'+self.subject+'_IsolationForest'+'.txt', 'rb') as file:
                self.modelers['IsolationForest'] = pickle.load(file)

        with open(self.model_path+'model_'+self.subject+'_IsolationForest' + '_params.txt', 'rb') as file:
            params = pickle.load(file)

        self.scaler = params['scaler']
        

    def save_models(self):
        with open(self.model_path+'model_'+self.subject+'_IsolationForest'+'.txt', 'wb') as file:
            pickle.dump(self.modelers['IsolationForest'], file)

        params = {}
        params['scaler'] = self.scaler

        with open(self.model_path+'model_'+self.subject+'_IsolationForest' + '_params.txt', 'wb') as file:
            pickle.dump(params, file)

    def make_predictions(self):
        isolation_forest = self.modelers['IsolationForest']

        data = self.source_db.read(self.subject+'_source', volume=1).get_data()
        time_col = data['Время']
        data = data[self.targets]

        data_scaled = self.scaler.transform(data)
        isolation_outliers = isolation_forest.predict(data_scaled)

        isolation_outliers = np.array([1 if label == -1 else 0 for label in isolation_outliers])
        
        df_result = pd.DataFrame(np.vstack([time_col, isolation_outliers]).T,
                                        columns=['time', 'outliers'])
            
        self.result_db.write_from_df(Table('Analitics_'+'clust_'+ self.subject, df_result),
                                            method='append')

       