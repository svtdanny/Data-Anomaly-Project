import pandas as pd

import numpy as np

import sqlite3
import sqlalchemy as db
from sqlalchemy.orm import sessionmaker

from Table import Table


class DataBase:
    def __init__(self, db_name):
        self.conn = sqlite3.connect(db_name)
        self.disk_engine = db.create_engine('sqlite:///' + db_name)
        self.cursor = self.conn.cursor()

        # create a configured "Session" class
        session = sessionmaker(bind=self.disk_engine)

        # create a Session
        self.session = session()

    def create_table(self, table_name, columns):
        self.cursor.execute(f'''CREATE TABLE IF NOT EXISTS {table_name}
                     ({','.join(columns)})''')

    def read(self, table_name, volume=0):
        df = pd.read_sql_query(f"select * from {table_name};", self.conn)

        if volume != 0:
            df = df.iloc[-volume:,:]

        return Table(table_name, df)

    def write_from_df(self, table, method = 'replace'):
        """
        :param df:
        :param table_name:
        :param method: 'replace'/'append' or'replace last'

        :return:
        """

        # Appending the results to table
        df = table.get_data()
        if method == 'replace last':
            # Пока будет так. Вообще, нужно добавить id и по нему
            batch_size = 1
            if self.get_table_length(table.get_name()) != -1:
                data = self.read(table.get_name()).get_data()
                len_df = len(df)
                print('!!!')
                print(df)
                data = data.iloc[:-len_df+batch_size, :]
                data = pd.concat([data, df], axis=0)
            else:
                data = df

            print('!!!')
            print(df)

            data.to_sql(table.get_name(), self.disk_engine, index=False, if_exists='replace')
            # DELETE FROM notes WHERE id = (SELECT MAX(id) FROM notes);
            pass
        else:
            df.to_sql(table.get_name(), self.disk_engine, index=False, if_exists = method)

    def get_table_length(self, table_name):
        self.cursor.execute(f'''SELECT count(*) FROM sqlite_master WHERE type='table' AND name='{table_name}';''')
        exists = self.cursor.fetchall()
        exists = int(exists[0][0])

        result = -1
        if exists == 1:
            self.cursor.execute(f'''SELECT count(*) FROM {table_name}''')
            result = self.cursor.fetchall()
            result = int(result[0][0])
        
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.session.close()



if __name__ == '__main__':
    """
    columns = ['datetime', 'timeseries1', 'timeseries2', 'timeseries3']

    db = DataBase('time_series.db')
    db.create_table('group1', columns)

    data = pd.DataFrame([pd.to_datetime('27.08.2000'), pd.to_datetime('28.08.2000'), pd.to_datetime('29.08.2000')],
                        columns=[columns[0]])
    data = pd.concat([data, pd.DataFrame(np.arange(9).reshape(3, 3), columns=columns[1:])], axis=1)

    db.write_from_df(Table('group1', data), method='append')
    """



    database = DataBase('time_series.db')

    print(database.read('group1', 3).get_data())
