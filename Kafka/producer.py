from kafka import KafkaProducer
import json
import pandas as pd
from numpy import generic

#for testing
import schedule
import time
import datetime as dt

#KAFKA_BROKER_URL = '192.168.1.138:9092'
KAFKA_BROKER_URL = 'localhost:9092'

def convert(o):
    if isinstance(o, generic): return o.item()
    raise TypeError

def send_kafka(message):
    producer = KafkaProducer(bootstrap_servers=KAFKA_BROKER_URL,
                                value_serializer=lambda v: json.dumps(v, default=convert).encode('utf-8'))

    print(message)
    producer.send('test', message)
    #producer.send("test", json.dumps(message, default=convert).encode('UTF-8'))
    producer.close()
    print('sended')


def send_test_from_df(data, obj, volume):
    global i
    cur_time = dt.datetime.now()

    to_zeros = ['Траффик', 'Нагрузка']

    df = data[i:i+volume].copy(deep=True)

    for col in to_zeros:
        df[col] = 0.
    df['Время'] = str(cur_time)
    df['Объект'] = obj

    for j in range(len(df)):
        message = dict(zip(df.columns, df.iloc[j]))
        #print(message)
        send_kafka(message)

    i+=volume
    #cur_time += dt.timedelta(minutes=2, seconds=30)

if __name__ == '__main__':
    #use_cols = ['Температура', 'Мощность', 'Траффик', 'Нагрузка', 'Объект', 'Время']
    use_cols = ['Температура', 'Мощность', 'Траффик', 'Нагрузка', 'Объект', 'Время']

    print('Object:')
    obj = input()
    print('Time to start:')

    #cur_time = input()
    #cur_time = pd.to_datetime(cur_time)


    
    df = pd.read_csv('../Source data/result_train.csv', sep=',', usecols=use_cols)

    i=0

    schedule.every(30).seconds.do(send_test_from_df, data=df, obj=obj, volume=1)
    
    #prepare(time_to_proc=to_datetime('2019-06-06 00:15:00'), mode = 'operate')
    #run(mode='predict')

    while True:
        schedule.run_pending()
        time.sleep(1)

    """
    df = pd.read_csv('../Source data/result_train.csv', sep=',', usecols=use_cols)[:10]
    df['Время'] = cur_time

    for i in range(len(df)):
        message = dict(zip(df.columns, df.iloc[i]))
        print(message)
        send_kafka(message)
    """