from Cores.Model import Modeler
from DBConnector import DataBase
import pandas as pd
from Table import Table
import numpy as np
import pandas as pd
from datetime import timedelta
import os.path


class CoreSeparated:
    def __init__(self, subject, targets, time_cols, source_db, result_db, n_predict, volume, interval):
        self.subject = subject
        self.targets = targets
        self.time_cols = time_cols

        self.n_predict = n_predict
        self.volume = volume
        self.interval = interval

        self.source_db = DataBase(source_db)
        self.result_db = DataBase(result_db)

        self.modelers = {}

        self.models_path = './models/separated/model_'+self.subject+'_'


    def fit_core(self, first=False):
        data = self.source_db.read(self.subject+'_source').get_data()
        print(data.iloc[:, :])
        for target in self.targets:
            modeler = Modeler(model_path=self.models_path+target+'.txt',
                              n_predict=self.n_predict,
                              volume=self.volume,
                              scaler='std',
                              drop_null=True)
 
            print(data[target])
            print(data[self.time_cols+['Время']])
            X, y = modeler.prepare_data(X=data[self.time_cols+['Время']],
                                        y=data[target],
                                        predict_phase=True)

            n_folds=6
            if len(X) <= self.n_predict + n_folds:
                print('Warning: Model wasn`t trained! Not enought clean data '+target)
                continue

            # Нужно для понимания, когда данные пришли
            time_col = pd.to_datetime(X['Время'])

            first_start = os.path.isfile(self.models_path + target + '.txt')

            X.drop(['Время'], axis=1, inplace=True)
            modeler.fit(X.iloc[:-self.n_predict, :], y[:-self.n_predict])
            self.modelers[self.subject+'_'+target] = modeler

            if not first_start:
                predictions, lower, upper = modeler.predict(X, interval=True)

                y_anom_search = y[: -self.n_predict].copy()

                anomalies = CoreSeparated.find_anomalies(y_anom_search,
                                                lower[:-self.n_predict], upper[:-self.n_predict])
                anom_dummy = np.full((self.n_predict,), np.NaN)

                anomalies_res = np.concatenate([anomalies, anom_dummy])
                y_res = np.concatenate([y_anom_search, anom_dummy])

                # Интерполяция времени на предсказанные периоды
                time_dummy = pd.date_range(start=time_col.iloc[-self.n_predict-1],
                                        periods=self.n_predict+1,
                                        closed='right',
                                        freq=self.interval)
                time_col = pd.concat([time_col.iloc[:-self.n_predict], pd.Series(time_dummy)], axis=0).astype(str)

                # Добавим первые значения показателя без предсказаний
                time_first = data['Время'][:self.volume]
                y_first = data[target][:self.volume]
                dummy_first = np.array([None for counter in range(self.volume)])

                time_col = pd.concat([pd.Series(time_first), time_col], axis=0).astype(str)
                y_res = np.concatenate([y_first, y_res], axis=0)
                predictions = np.concatenate([y_first, predictions], axis=0)
                lower = np.concatenate([dummy_first, lower], axis=0)
                upper = np.concatenate([dummy_first, upper], axis=0)
                anomalies_res = np.concatenate([dummy_first, anomalies_res], axis=0)
                ##

                df_result = pd.DataFrame(np.vstack([time_col, y_res, predictions , lower,
                                                    upper, anomalies_res]).T,
                                        columns=['time', 'actual', 'predictions', 'lower', 'upper', 'anomalies'])
                self.result_db.write_from_df(Table('Analitics_'+'sep_'+self.subject+'_'+target, df_result),
                                            method='replace')

    @staticmethod
    def find_anomalies(y_true, lower, upper):
        anomalies = np.array([np.NaN]*len(y_true))
        anomalies[y_true < lower] = y_true[y_true<lower]
        anomalies[y_true > upper] = y_true[y_true>upper]

        return anomalies

    def load_models(self):
        """
        ! load None if model doesn`t exists
        """
        for target in self.targets:
            if os.path.isfile(self.models_path + target + '.txt'):
                modeler = Modeler(model_path=self.models_path + target + '.txt',
                                n_predict=self.n_predict,
                                volume=self.volume,
                                scaler='std',
                                drop_null=True)
                modeler.load_model_params()
            else:
                #if model doesn`t exists yet
                modeler = None
            
            self.modelers[self.subject + '_' + target] = modeler

    def make_predictions(self):
        for target in self.targets:
            modeler = self.modelers[self.subject+'_'+target]
            if modeler is None:
                #model doesn`t exists yet
                continue

            volume = modeler.get_volume()

            batch_volume = 1
            data = self.source_db.read(self.subject+'_source', volume=volume+self.n_predict).get_data()

            X, y = modeler.prepare_data(X=data[self.time_cols+['Время']],
                                        y=data[target],
                                        predict_phase=True, drop_null=False)

            time = pd.Series(pd.to_datetime(X['Время']))

            start_time = time.iloc[-self.n_predict-batch_volume]
            time = pd.Series(pd.date_range(start=start_time,
                                           periods=self.n_predict+batch_volume,
                                           closed='left',
                                           freq=self.interval))
            time = time.astype(str)

            X.drop(['Время'], axis=1, inplace=True)

            predictions, lower, upper = modeler.predict(X, interval=True)

            anomalies = CoreSeparated.find_anomalies(y, lower, upper)

            #send data to outer messager

            actual = data[target][-batch_volume:]
            actual_dummy = np.array([np.NaN for i in range(len(predictions) - batch_volume)])
            actual = np.concatenate([actual, actual_dummy])

            df_result = pd.DataFrame(np.vstack([time, actual, predictions, lower,
                                                upper, anomalies]).T,
                                     columns=['time', 'actual', 'predictions', 'lower', 'upper', 'anomalies'])
            self.result_db.write_from_df(Table('Analitics_'+'sep_'+self.subject+'_'+target, df_result), method='replace last')
