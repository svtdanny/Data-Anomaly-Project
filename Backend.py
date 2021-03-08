from Engine import Engine
from DataPreparator import DataPreparator
from pandas import to_datetime
import schedule
from datetime import timedelta
import time
import datetime as dt


subjects = ['Главный', 'Вспомогательный']
#avg_window = '600S'
avg_window = '60S'

def run(mode='fit', first=False):
    """
    mode - (str) - 'fit'/'predict'
    """
    global subjects
    # ['Температура', 'Мощность', 'Траффик', 'Нагрузка']
    targets = ['Температура', 'Мощность', 'Траффик', 'Нагрузка']
    time_cols = ['hour', 'weekday', 'is_weekend']

    engine_params = {
        "Главный": [targets, time_cols],
        "Вспомогательный": [targets, time_cols],

    }

    engine = Engine(dict_params = engine_params,
                    interval=avg_window,
                    n_predict=1,
                    volume=10)

    if mode == 'fit':
        engine.fit_engine(first)
    elif mode == 'predict':
        engine.make_predictions()
    print('Engine successfully fitted')

    for sub in subjects:
        for target in targets:
            if engine.SimpleCores[sub].result_db.get_table_length('Analitics_'+'sep_'+sub+'_'+target) != -1:
                print(sub)
                print(sub+'_'+target)
                print(engine.SimpleCores[sub].result_db.read('Analitics_'+'sep_'+sub+'_'+target).get_data().iloc[-50:, :])


def prepare(avg_window, time_to_proc=None, mode='operate'):
    """
    time_to_proc - time to current processing
    mode - 'start'/'operate'
    """
    global subjects

    use_cols = ['Температура', 'Мощность', 'Траффик', 'Нагрузка', 'Объект', 'Время']

    if mode == 'start':
        data_path = 'result_train.csv'
    elif mode == 'operate':
        data_path = 'result_test.csv'
    else:
        raise ValueError('mode is not correct')

    preparator = DataPreparator('BaseDB.db',
                                use_cols=use_cols,
                                data_path=data_path,
                                time_col='Время',
                                cols_to_average=['Температура', 'Нагрузка'],
                                entities_to_use=subjects,
                                group_by='Объект',
                                avg_window=avg_window,
                                decode_UNIX_time=False,
                                prepare_time=True,
                                mode=mode,
                                time_to_proc=time_to_proc,
                                from_outer=True)

    result, time_cols = preparator.preprocess_data()
    preparator.load_data_to_db(result)


def update_job(simple_mode=False):
    #global time_to_proc
    global delta

    time_to_proc = dt.datetime.now()
    time_to_proc = time_to_proc - dt.timedelta(seconds=time_to_proc.second % delta.total_seconds())

    prepare(time_to_proc=time_to_proc, mode='operate', avg_window=avg_window)
    if not simple_mode:
        run(mode='predict')
    time_to_proc = time_to_proc+delta

simple_mode = False

if __name__ == '__main__':
    time_to_proc = dt.datetime.now()
    delta=dt.timedelta(seconds=int(avg_window[:-1]))
    time_to_proc = time_to_proc - dt.timedelta(seconds=time_to_proc.second % delta.total_seconds())

    prepare(mode='start', time_to_proc=time_to_proc, avg_window=avg_window)

    time_to_proc += delta

    if not simple_mode:
        run(mode='fit', first=True)



    schedule.every(1).minutes.at(":10").do(update_job, simple_mode=False)
    if not simple_mode:
        schedule.every(3).minutes.at(":20").do(run, mode='fit')

    #prepare(time_to_proc=to_datetime('2019-06-06 00:15:00'), mode = 'operate')
    #run(mode='predict')

    while True:
        schedule.run_pending()
        time.sleep(1)



"""
['pktTotalCount',
                'octetTotalCount', 'flowStart', 'flowDuration',
                'avg_piat', 'f_pktTotalCount', 'f_octetTotalCount', 'b_pktTotalCount', 'b_octetTotalCount', 'web_service'
                ]
"""