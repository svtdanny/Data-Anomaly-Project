import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.linear_model import LassoCV, RidgeCV
from os.path import isfile


class Modeler:

    def __init__(self, model_path, n_predict=1, volume=10, scaler='std', drop_null=False):
        self.model_path = model_path
        if isfile('./models/' + self.model_path):
            with open('./models/' + self.model_path, 'rb') as file:
                self.model = pickle.load(file)

        if scaler == 'std':
            self.scaler = StandardScaler()
        else:
            self.scaler = None

        self.lag_start = n_predict
        self.lag_end = self.lag_start + volume - 1
        self.volume = volume
        self.data_size_predict = 2*(volume-1)+1

        self.drop_null = drop_null

        # Инициализация нулем
        self.mae = 0
        self.scale = 1
        self.mae_deviation = 0

    def get_lags(self):
        return self.lag_start, self.lag_end

    def get_data_size_predict(self):
        return self.data_size_predict

    def get_volume(self):
        return self.volume

    def get_drop_null(self):
        return self.drop_null

    def predict(self, X, interval=False):
        """
        param X - Признаки для модели
        param y - Исторические данные пердсказывемого параметра. Нужен только для создания лагов
        param interval: Подсчет доверительных интервалов

        return
        Если interval == True
        prediction, lower interval, upper interval

        Если interval == False
        prediction, 0, 0
        """

        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)

        if interval:
            lowers = predictions - (self.mae + self.scale * self.mae_deviation)
            uppers = predictions + (self.mae + self.scale * self.mae_deviation)

            return np.vstack([predictions, lowers, uppers])

        else:
            return np.vstack([predictions, 0, 0])

    def prepare_data(self, X, y, predict_phase=False, drop_null = True):
        assert (X.shape[0] == y.shape[0])
        assert (len(y.shape) == 1)

        target = pd.DataFrame(y)
        target.columns = ['target']
        data = pd.concat([X, target], axis=1)

        if drop_null:
            data = data[data.loc[:, 'target'] != 0]
            data.reset_index(drop=True, inplace=True)

        # lags of series
        if predict_phase:
            for i in range(0, (self.lag_end - self.lag_start + 1)):
                data["lag_{}".format(i + self.lag_start)] = data['target'].shift(i)
            data.dropna(inplace=True, subset=[col for col in data.columns if col != 'target'])
            data['target'] = data['target'].shift(-self.lag_start)
            data['Время'] = data['Время'].shift(-self.lag_start)

        else:
            for i in range(self.lag_start, self.lag_end+1):
                data["lag_{}".format(i)] = data['target'].shift(i)
            data.dropna(inplace=True)

        data.reset_index(drop=True, inplace=True)

        return data.drop(['target'], axis=1), data['target']

    @staticmethod
    def outlier_detect_mean_std(data, col, threshold=3, info=False):
        """Определяем выбросы по правилу threshold сигм"""
        Upper_fence = data[col].mean() + threshold * data[col].std()
        Lower_fence = data[col].mean() - threshold * data[col].std()
        params = (Upper_fence, Lower_fence)
        tmp = pd.concat([data[col] > Upper_fence, data[col] < Lower_fence], axis=1)
        outlier_index = tmp.any(axis=1)
        if info:
            if (len(outlier_index.value_counts()) < 2):
                print('Количество выбросов в данных:', 0)
            else:
                print('Количество выбросов в данных:', outlier_index.value_counts()[1])
                print('Доля выбросов:', outlier_index.value_counts()[1] / len(outlier_index))

        return outlier_index, params

    def load_model_params(self):
        with open(self.model_path, 'rb') as file:
            self.model = pickle.load(file)

        with open(self.model_path[:-4] + '_' + 'params.txt', 'rb') as file:
            params = pickle.load(file)

        self.scaler = params['scaler']
        self.mae = params['mae']
        self.scale = params['scale']
        self.mae_deviation = params['mae_deviation']

    def save_model_params(self):
        with open(self.model_path, 'wb') as file:
            pickle.dump(self.model, file)

        params = {}
        params['scaler'] = self.scaler
        params['mae'] = self.mae
        params['scale'] = self.scale
        params['mae_deviation'] = self.mae_deviation

        with open(self.model_path[:-4] + '_' + 'params.txt', 'wb') as file:
            pickle.dump(params, file)

    def fit(self, X, y):
        #print(len(X))
        #print(X)
        #X_train_scaled, y_train = self.prepare_data(X, y, predict_phase=False, drop_null=self.drop_null)
        y_train = y
        X_train_scaled = self.scaler.fit_transform(X)

        outlier_index, _ = Modeler.outlier_detect_mean_std(pd.DataFrame(y_train, columns=['y']), col='y', threshold=3, info=False)
        print(outlier_index)
        if len(outlier_index) !=0: 
            X_train_scaled = X_train_scaled[~outlier_index]
            y_train = y_train[~outlier_index]

        tscv = TimeSeriesSplit(n_splits=5)

        self.model = RidgeCV(cv=tscv, alphas=10 ** np.linspace(-3, 0.5, num=10), scoring="neg_mean_absolute_error")
        self.model.fit(X_train_scaled, y_train)

        cv = cross_val_score(self.model, X_train_scaled, y_train,
                             cv=tscv,
                             scoring="neg_mean_absolute_error")
        self.mae = cv.mean() * (-1)
        self.mae_deviation = cv.std()

        self.save_model_params()
