from Cores.CoreSeparated import CoreSeparated
from Cores.CoreClust import CoreClust

from datetime import datetime

from DBConnector import DataBase

class Engine:
    def __init__(self, dict_params, interval, n_predict, volume):
        """
        dict_params - словарь subject: [targets, time_cols]
        interval
        """
        self.subjects = dict_params.keys()

        self.ClustCores = {}
        self.SimpleCores = {}

        for subject in list(dict_params.keys()):
            targets = dict_params[subject][0]
            time_cols = dict_params[subject][1]

            self.SimpleCores[subject] = CoreSeparated(subject=subject,
                                       targets=targets,
                                       time_cols=time_cols,
                                       source_db='BaseDB.db',
                                       result_db='BaseDB.db',
                                       n_predict=n_predict,
                                       volume=volume,
                                       interval=interval
                                       )

            self.ClustCores[subject] = CoreClust(subject=subject,
                                        targets=targets,
                                        time_cols=time_cols,
                                        source_db='BaseDB.db',
                                        result_db='BaseDB.db'
                                        )
                                       

    def fit_engine(self, first=False):
        for subject in list(self.subjects):
            self.SimpleCores[subject].fit_core(first)

            self.ClustCores[subject].fit_core(first)


    def make_predictions(self):
        for subject in list(self.subjects):
            self.SimpleCores[subject].load_models()
            self.SimpleCores[subject].make_predictions()

            self.ClustCores[subject].load_models()
            self.ClustCores[subject].make_predictions()

        with open('./fit_logs.txt', 'a') as file:
            file.write('\n')
            file.write(str(datetime.now()))
