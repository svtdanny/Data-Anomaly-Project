from DBConnector import DataBase
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from Table import Table

from Kafka.KafkaConnector import KafkaConnector

class DataPreparator:
    def __init__(self, destination_db, use_cols, data_path, time_col, cols_to_average, entities_to_use, group_by, avg_window='300S',
                        decode_UNIX_time=False, prepare_time=False, trig_prepare_time=False, mode='load', from_outer=False, time_to_proc=None):

        self.destination_db = DataBase(destination_db)
        self.use_cols = use_cols
        self.data_path = data_path
        self.from_outer = from_outer
        self.time_to_proc = time_to_proc
        self.mode = mode

        self.time_col = time_col
        self.cols_to_average = cols_to_average
        self.group_by = group_by
        self.avg_window = avg_window

        self.entities_to_use = entities_to_use

        self.decode_UNIX_time = decode_UNIX_time
        self.prepare_time = prepare_time
        self.trig_prepare_time = trig_prepare_time

        self.data = pd.DataFrame()

    def upload_data(self):
        if self.from_outer:
            #Здесь запуск скрипта загрузки данных извне (например чтение из Kafka)
            #df = pd.read_csv('./Source data/'+self.data_path, sep=',', usecols=self.use_cols)
            kafka_conn = KafkaConnector()
            df = kafka_conn.load_data()
        else:
            df = self.destination_db.read(self.data_path).get_data()

        if len(df) == 0:
            df = pd.DataFrame([], columns=self.use_cols)
            #self.data = df
            #return

        if self.decode_UNIX_time:
            # приводим время к читаемому формату:
            df[self.time_col] = df[self.time_col].apply(lambda x: datetime.fromtimestamp(x))
        else:
            df[self.time_col] = pd.to_datetime(df[self.time_col])

        if self.time_to_proc and self.mode == 'operate':
            delta = timedelta(seconds=int(self.avg_window[:-1]))
            thr_low = self.time_to_proc-delta <= df[self.time_col]
            thr_up = self.time_to_proc > df[self.time_col]
            print('CURRENT TIME')
            print(self.time_to_proc)

            df = df[thr_low & thr_up]
            # Здесь удалить все то, что было раньше из источника, если необходимо

        self.data = df

    def preprocess_data(self):
        """
        df - исходные данные (только с нужными колонками)
        time_col - колонка времени
        cols_to_average - Колонки, для которых надо посчитать среднее значение по периоду (для остальных - сумма по периоду)
        entities_to_use - Значения колонки источников, необходимые для рассмотрения
        group_by - Колонка с источниками
        avg_window - Период, по который нужно сужать данные
        decode_UNIX_time - НАдо ли декодировать время из формата времени UNIX
        prepare_time - Нужно ли добавлять признаки час, день недели, выходной день
        trig_prepare_time - Кодировка времени по sin/cos


        return
        - dict - {источник:данные}
        - list - название колонок временных признаков
        """
        self.upload_data()
        df = self.data

        # Добавим колонку с "1", чтобы после усреднения знать количество соединений за каждый период времени
        df['ConnectionCount'] = np.ones((len(df), 1), dtype=int)

        # Словарь для записи результатов
        result = {}

        for entity in self.entities_to_use:
            print(df[df[self.group_by] == entity])
            df_res = df[df[self.group_by] == entity]
            print(self.group_by)
            print(entity)
            print(df)
            if (len(df_res) == 0):
                
                if not (self.time_to_proc):
                    #self.time_to_proc = pd.to_datetime('2018-08-31 15:30:00')
                    raise Exception('Start up error: ', 'start system without data in buffer and without specifying time_to_proc')

                a = np.zeros(shape=(1, len(df.columns)), dtype=int)
                df_res = pd.DataFrame(a, columns=df_res.columns, index=[0])
                df_res.loc[0, self.time_col] = pd.to_datetime(self.time_to_proc) - timedelta(seconds=
                                                                                             int(self.avg_window[:-1]))
                df_res.drop([self.group_by], axis=1, inplace=True)
                print(df_res[[self.time_col]])

            df_res = df_res.resample(self.avg_window, on=self.time_col).sum()


            df_res[self.time_col] = df_res.index
            df_res.reset_index(drop=True, inplace=True)

            if self.prepare_time or self.trig_prepare_time:
                # datetime features
                df_res["hour"] = df_res[self.time_col].apply(lambda x: x.hour)
                df_res["weekday"] = df_res[self.time_col].apply(lambda x: x.weekday())
                df_res['is_weekend'] = df_res[self.time_col].apply(lambda x: x.weekday()).isin([5, 6]) * 1

                if self.trig_prepare_time:
                    hours_in_day = 24
                    weekdays_in_week = 7

                    df_res['sin_hour'] = np.sin(2 * np.pi * df_res["hour"] / hours_in_day)
                    df_res['cos_hour'] = np.cos(2 * np.pi * df_res["hour"] / hours_in_day)

                    df_res['sin_weekday'] = np.sin(2 * np.pi * df_res["weekday"] / weekdays_in_week)
                    df_res['cos_weekday'] = np.cos(2 * np.pi * df_res["weekday"] / weekdays_in_week)

                    df_res.drop(['hour'], axis=1, inplace=True)
                    df_res.drop(['weekday'], axis=1, inplace=True)

            for col in self.cols_to_average:
                print('!!!!!!!')
                print(df_res['ConnectionCount'])
                df_res[col] = (df_res[col] / df_res['ConnectionCount'])
                df_res[[col]] = df_res[[col]].fillna(value=0)
               
            result[entity] = df_res

        if self.trig_prepare_time:
            result_time_cols = [self.time_col, 'sin_hour', 'cos_hour', 'sin_weekday', 'cos_weekday', 'is_weekend']
        elif self.prepare_time:
            result_time_cols = [self.time_col, 'hour', 'weekday', 'is_weekend']
        else:
            result_time_cols = [self.time_col]

        return result, result_time_cols

    def load_data_to_db(self, result_dict):
        for subject in list(result_dict.keys()):
            print(result_dict[subject])
            self.destination_db.write_from_df(Table(subject+'_source',
                                                    result_dict[subject]),
                                              method='append')
