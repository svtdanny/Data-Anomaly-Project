# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from math import trunc,sqrt,ceil
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
import datetime as dt
import os

from dash_extensions import Download
from dash_extensions.snippets import send_bytes
import xlsxwriter

from DBConnector import DataBase

import dash_daq as daq

from .. app import app

# get relative data folder
#PATH = pathlib.Path(__file__).parent

subjects = ['Главный', 'Вспомогательный']
targets = ['Температура', 'Мощность', 'Траффик', 'Нагрузка']
Dataframes = {}
ClustDataframes = {}
ActualDataframes = {}

def upload_separate_data():
    db = DataBase('BaseDB.db')
    res = {}

    for sub in subjects:
        for target in targets:
            if (db.get_table_length('Analitics_'+'sep_'+sub + '_' + target)==-1):
                print('Table '+ 'Analitics_'+'sep_'+sub + '_' + target + ' doen`t exists yet!')
            else:
                res[sub+'_'+target] = db.read('Analitics_'+'sep_'+sub + '_' + target).get_data()[-150:]
                print(res[sub+'_'+target])

    return res

def upload_clust_data():
    db = DataBase('BaseDB.db')
    res = {}

    for sub in subjects:
        #Необходимо отрезать первые 10,
        res[sub] = db.read('Analitics_'+'clust_'+sub).get_data()[-150:]
        print('Clust ', sub)
        print("@@@")
        #print(res[sub][-100:-50])
        print(res[sub][:50])
        print('|||')
        print(res[sub][-50:])

    return res

def upload_simple_data():
    db = DataBase('BaseDB.db')
    res = {}

    for sub in subjects:
        sub_df = db.read(sub+'_source').get_data()[-150:]
        #print(sub_df.columns)
        for target in targets:
            res[sub+'_'+target] = sub_df[['Время', target]]
            res[sub+'_'+target].columns = ['time', 'actual']

            #print(res[sub+'_'+target])

    return res

def upload_data(simple_mode=False):
    global Dataframes
    global ClustDataframes
    global ActualDataframes

    if not simple_mode:
        ActualDataframes = upload_simple_data()
        Dataframes = upload_separate_data()
        ClustDataframes = upload_clust_data()
    else:
        ActualDataframes = upload_simple_data()


upload_data()

def make_vrect_shapes(periods, color="lightgray"):
    res=[]
    print('periods')
    print(periods)
    for x0, x1 in periods:
        print(x0,x1)
        res.append(dict(
                type="vrect",
                x0=x0,
                y0=0,
                x1=x1,
                y1=1,
                fillcolor=color,
                opacity=0.6,
                yref='paper',
                line={
                    'width': 0
                    },
                layer="below"
            ))

    return res


def make_clust_monitoring_anomalies(subject):
    outliers = ClustDataframes[subject]['outliers']

    print('Clust making')
    print(ClustDataframes[subject])


    ClustData_lag = ClustDataframes[subject][outliers.shift(-1) != 0].iloc[:-1]
    ClustData_lag = ClustData_lag['time'].values
    ClustData = ClustDataframes[subject].iloc[1:][outliers.iloc[1:] != 0]
    ClustData = ClustData['time'].values

    print('LAGS')
    print(ClustData_lag)

    print('CURR')
    print(ClustData)

    intervals=np.vstack([ClustData_lag,  ClustData]).transpose(1,0)
    res = make_vrect_shapes(intervals)
    #print(intervals[0])
    return res
 

def make_ident(k, i, j, size=15):
    return html.Div(id='id-box'+str(k)+str(i)+str(j),
                    style={'width': '7%'},
                    className="six columns",
                    children=[
                        daq.Indicator(
                            id='my-daq-indicator'+str(k)+str(i)+str(j),
                            value=True,
                            color="#00cc96",
                            size=size,
                        )
                    ])


def make_island(land, N_elems):
    rows = trunc(sqrt(N_elems))
    cols = [ceil(N_elems*1./rows) for i in range(rows-1)]+[N_elems-(rows-1)*ceil(N_elems*1./rows)]
    return html.Div(
            id='id-box-land'+str(land),
            className="six columns inner-row",
            style={'width': '20%'},
            children=[build_major_title(str(land))]+[
                html.Div(
                    id='id-box-row'+str(land)+str(i),
                    style={'width': '100%'},
                    className="six columns",
                    children=[
                        make_ident(land,i,j) for j in range(cols[i])
                        ]
                    ) for i in range(rows)
            ],
        )


def make_anomalies_data(subjects, targets):
    result = []

    for i, subject in enumerate(subjects):
        for target in targets:
            if (subject + '_' + target) in Dataframes:
                print('Sep analitics starts working')
                data = Dataframes[subject + '_' + target][~pd.isna(Dataframes[subject + '_' + target]['anomalies'])]
                
                res = dict(
                        x=data['time'],
                        y=(len(subjects) - i) * np.ones((len(data),)),
                        text='warning: ' + target + ' actual: '+ data['actual'].astype(str) +
                            ' predicted: ' + data['predictions'].map('{:,.2f}'.format).astype(str),
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                            },
                        name=subject + ' ' + target
                    )

                result.append(res)


        # Clust data
        ClustData = ClustDataframes[subject][ClustDataframes[subject]['outliers'] != 0]
        res = dict(
                x=ClustData['time'],
                y=(len(subjects) - i) * np.ones((len(ClustData),)),
                text='warning: ' + 'context anomaly',
                mode='markers',
                opacity=0.7,
                marker={
                    'size': 15,
                    'line': {'width': 0.5, 'color': 'white'}, 
                    'symbol': 'diamond'
                    },
                name=subject + ' ' + target
                )   

        result.append(res)


    return result


def make_monitoring_data(subject, target, simple_mode=False):
    result = []

    if (subject+'_'+target) in Dataframes:
        result = result + [dict(
                x=Dataframes[subject+'_'+target]['time'],
                y=Dataframes[subject+'_'+target][col],
                line=dict(width=1, dash='dash', color='#f4584f'),
                name='Границы доверительного интервала',

                ) for col in ['lower', 'upper']]

        result.append(dict(
            x=Dataframes[subject + '_' + target]['time'],
            y=Dataframes[subject + '_' + target]['predictions'],
            mode='Scatter',
            line=dict(color = '#f4584f', width=4, opacity=0.7),
            name='Предсказание'
        ))
    
    result.append(dict(
        x=ActualDataframes[subject+'_'+target]['time'],
        y=ActualDataframes[subject+'_'+target]['actual'],
        mode='Scatter',
        line=dict(color='#60bcb6', width=4, opacity=0.7),
        name='Показатель'
    ))

    if (subject+'_'+target) in Dataframes:
        result.append(dict(
            x=Dataframes[subject + '_' + target]['time'],
            y=Dataframes[subject + '_' + target]['anomalies'],
            mode='markers',
            marker=dict(color='#323232', size=13, opacity=0.7),
            name='Аномалия'
        ))

    


    return result


def make_text_log(subject, width, start_date, end_date, log_len=7, simple_mode=True):
    text_log_df = pd.DataFrame()
    for target in targets:
        if (subject + '_' + target) not in Dataframes:
            continue

        data_to_merge = Dataframes[subject + '_' + target].copy(deep=True)
        data_to_merge['target'] = target
        data_to_merge = data_to_merge.dropna()

        text_log_df = pd.concat([text_log_df, data_to_merge], axis=0, ignore_index=True)

    if len(text_log_df) == 0:
        return html.Div(
                    id="logs_tape_" + subject,
                    className="six columns inner-row",
                    style={'width': width},
                    children=[build_major_title(subject)],
                    )

    text_log_df.dropna(inplace=True)
    text_log_df['time'] = pd.to_datetime(text_log_df['time'])
    text_log_df.sort_values(by=['time'], inplace=True)

    logs_tape = []
    timestamps = text_log_df['time']

    timestamps = timestamps[start_date <= timestamps][timestamps < end_date]
    timestamps = timestamps.unique()
    timestamps.sort()
    timestamps = timestamps[::-1]

    for timestamp in timestamps:
        df_timestamp = text_log_df[text_log_df['time'] == timestamp]

        text = []
        for i in range(len(df_timestamp)):
            if not simple_mode:
                text.extend([
                    build_medium_title(df_timestamp.iloc[i, :]['target']),
                    build_medium_title(str(pd.to_datetime(timestamp))),
                    f"""Показатель: {df_timestamp.iloc[i, :]['actual']:0.2f}""",
                    html.Br(),
                    f"""Предсказание: {df_timestamp.iloc[i, :]['predictions']:0.2f}"""])
            else:
                text.extend([
                    build_medium_title(df_timestamp.iloc[i, :]['target']),
                    build_medium_title(str(pd.to_datetime(timestamp))),
                    f"""Показатель: {df_timestamp.iloc[i, :]['actual']:0.2f}"""])

        logs_tape.append(
            html.Div(
                children=[
                    html.H6(children=text)
                ]
            )
        )

    return html.Div(
        id="logs_tape_" + subject,
        className="six columns inner-row",
        style={'width': width},
        children=[build_major_title(subject)] + logs_tape[:log_len],
        )


def make_logs_panel(start_date=datetime(2000, 1, 1), end_date=datetime.now()):
    panel = []
    for subject in subjects:
        panel.append(make_text_log(subject, '40%', start_date, end_date))

    return panel


def build_major_title(title):
    return html.P(className="graph-title", children=title)


def build_company_name(title):
    return html.P(className="company-name", children=title)


def build_medium_title(title):
    return html.P(className="title-middle", children=title)


@app.callback(Output('all_anomalies_on_groups', 'figure'),
              [Input('date-picker-range', 'start_date'),
               Input('date-picker-range', 'end_date'),
               Input('synch_output', 'data')
               ])
def update_anomalies_on_groups(start_date, end_date, _):
    data = make_anomalies_data(subjects, targets)
    tickvals = list(range(1, len(subjects)+1))[::-1]
    ticktext = [s[:5] for s in subjects]
    return{'data': list(data),
           'layout': dict(
                xaxis={'title': 'Время', 'range': [start_date, end_date]},
                yaxis={'title': 'Сервисы', 'tick0': 0, 'dtick': 1, 'tickmode':'array', 'tickvals':tickvals, 'ticktext':ticktext},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                showlegend=False,
                hovermode='closest'
                )
           }


@app.callback(Output('monitoring-data-graph', 'figure'),
              [Input('subject-dropout', 'value'),
               Input('target-dropout', 'value'),
               Input('date-picker-range', 'start_date'),
               Input('date-picker-range', 'end_date'),
               Input('synch_output', 'data')
               ])
def update_monitoring(subject, target, start_date, end_date, _):
    data = make_monitoring_data(subject, target)
    return {'data': list(data),
            'layout': dict(
                shapes= make_clust_monitoring_anomalies(subject),
                xaxis={'title': 'Время', 'range': [start_date, end_date]},
                yaxis={'title': ''},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest',
                )
            }


@app.callback(Output('logs_panel', 'children'),
              [Input('date-picker-range', 'start_date'),
               Input('date-picker-range', 'end_date'),
               Input('synch_output', 'data')
               ])
def update_text_logs(start_date, end_date, _):
    return make_logs_panel(start_date, end_date)


@app.callback(Output('synch_output', 'data'),
              [Input('interval-component', 'n_intervals')],
              [State('synch_output', 'data')])
def update_metrics(_, data):
    with open('./fit_logs.txt', 'rb') as file:
        
        last_date = pd.to_datetime(file.readlines()[-1].decode("utf-8"))
        last = int(last_date.timestamp())
        
        if  data and last <= data['num']:
            print(0)
            return dash.no_update
        else:
            print(1)
            print(last_date)
            upload_data()
            
            data = {'num': last}

            return data

@app.callback(Output("download", "data"), [Input("btn", "n_clicks")])
def generate_report(n_nlicks):
    ctx = dash.callback_context
    if ctx.triggered[0]['prop_id'].split('.')[0] != 'btn':
        print('Button wasn`t clicked!')
        return dash.no_update

    report = pd.read_excel('./Report/Отчет по качеству сети.xlsx', sheet_name=None)

    def to_xlsx(bytes_io):
        writer = pd.ExcelWriter(bytes_io, engine="xlsxwriter")
        keys = list(report.keys())
        for key in keys[:2]:
            report[key].to_excel(writer, sheet_name=key)
        writer.save()

    return send_bytes(to_xlsx, filename='Отчет по качеству сети.xlsx')
    


tickvals = list(range(1, len(subjects)+1))[::-1]
ticktext = [s[:5] for s in subjects]

layout = html.Div(
    style={'background': '#f2f5fa'},
    id="bottom-row",
    children=[
    html.Div(
        className="row",

        children=[
            dcc.Store(id='synch_output'),
            dcc.Interval(
                id='interval-component',
                interval=1*3000,  # in milliseconds
                n_intervals=0
            ),
            html.Div(
                id="dropouts-container",
                className="six columns",

                children=[
                    html.Div(
                        id="Company_name",
                        className="six columns inner-row",
                        style={'width': '40%'},

                        children=[
                            html.Div(children=build_company_name('Alias')),

                        ],
                    ),
                    html.Div(
                        id="info2",
                        className="six columns",
                        style={'width': '10%'},

                        children=[
                            html.Div(children='''Температура'''),
                            build_major_title('13 C'),
                        ],
                    ),

                    html.Div(
                        id="well-production-container",
                        className="six columns",
                        style={'width': '10%'},
                        children=[
                            html.Div(children='''Влажность'''),
                            build_major_title('43 %'),
                        ],
                    ),

                    html.Div(
                        id="info1",
                        className="six columns",
                        style={'width': '30%', 'float': 'right'},

                        children=[
                            html.Div(children='''Общая нагрузка'''),
                            daq.GraduatedBar(
                                id='my-daq-graduatedbar',
                                value=5,
                            )
                        ],
                    ),
                ],
            ),

            html.Div(
                id="info-container",
                className="six columns",

                children=[
                    html.Div(
                        id="dropout1",
                        className="six columns inner-row",
                        style={'width': '35%'},
                        children=[
                            html.Div(children='''Период'''),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                start_date=datetime.now() - pd.to_timedelta('2H'),
                                display_format='DD-MM-YYYY',
                                end_date=datetime.now() + pd.to_timedelta('0.5H'),
                            ),
                        ],
                    ),
                    html.Div(
                        id="subject-dropout-box",
                        className="six columns",
                        style={'width': '20%'},
                        children=[
                            html.Div(children='''Сервис'''),
                            dcc.Dropdown(
                                id="subject-dropout",
                                options=[
                                    {
                                        'label': subject, 'value': subject
                                    } for subject in subjects
                                ],
                                value=subjects[0],
                            ),
                        ],
                    ),

                    html.Div(
                        id="target-dropout-box",
                        className="six columns",
                        style={'width': '20%'},
                        children=[
                                html.Div(children='''Показатель'''),
                                dcc.Dropdown(
                                    id = 'target-dropout',
                                    options=[
                                        {'label': target, 'value': target} for target in targets
                                    ],
                                    value=targets[0],
                                ),

                        ],
                    ),

                    html.Div(
                        className="six columns",
                        style={'width': '15%'},
                        children=[
                            html.Br(),
                            html.Button("Скачать отчет", id="btn"), Download(id="download")
                            ]
                        ),

                ],
            ),

        ],
    ),

    html.Br(),

    html.Div(
        className="row",
        id="bottom-row2",

        children=[
            
            html.Div(
                id="islands",
                className="six columns",

                children=[
                    dcc.Graph(
                        id='all_anomalies_on_groups',
                        figure={
                            'data': make_anomalies_data(subjects, targets),
                            'layout': dict(
                                xaxis={'title': 'Время'},
                                yaxis={'title': 'Сервисы', 'tick0': 0, 'dtick': 1, 'tickmode': 'array',
                                       'tickvals': tickvals, 'ticktext': ticktext},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                showlegend=False,
                                hovermode='closest'
                                )
                        }
                    )
                ],
                
            ),
            

            html.Div(
                id="Capacity",
                className="six columns",

                children=[
                    dcc.Graph(
                        id='monitoring-data-graph',
                        figure={
                            'data': make_monitoring_data(subjects[0], targets[0]),
                            
        
                            'layout': dict(
   
                                xaxis={'title': 'GDP Per Capita'},
                                yaxis={'title': 'Life Expectancy'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest',

                                ),

                        }
                    )
                ],
            ),
        ],
    ),

    html.Br(),

    html.Div(
        className="row",
        id="bottom-row3",
        children=[
            
            html.Div(
                id="islands1",
                className="six columns",

                children=[
                    html.Div(
                        id="subject-dropout-box3",
                        className="six columns inner-row",
                        style={'width': '26%'},
                        children=[
                            html.Div(children='''Объект'''),
                            dcc.Dropdown(
                                id="subject-dropout3",
                                options=[
                                    {
                                        'label': subject, 'value': subject
                                    } for subject in subjects
                                ],
                                value=subjects[0],
                            ),
                        ],
                    ),

                    html.Div(
                        id="target-dropout-box3",
                        className="six columns",
                        style={'width': '26%'},
                        children=[
                                html.Div(children='''Событие'''),
                                dcc.Dropdown(
                                    id = 'target-dropout3',
                                    options=[
                                        {'label': target, 'value': target} for target in targets
                                    ],
                                    value=targets[0],
                                ),

                        ],
                    ),

                    html.Div(
                        id="dropout31",
                        className="six columns",
                        style={'width': '16%'},
                        children=[
                            html.Div(children='''Время'''),
                            dcc.DatePickerSingle(
                            id="dt1",              
                            ),
                        ],
                    ),

                    html.Div(
                        id="dropout41",
                        className="six columns",
                        style={'width': '26%'},
                        children=[
                            html.Br(),
                            dcc.Input(type="text",
                            id="SelectionFrom",   
                            value = '12:45'           
                            ),
                        ],
                    ),

                    dcc.Textarea(
                        id='textarea-example',
                        value='',
                        style={'width': '100%', 'height': 100},
                    ),

                    html.Div(
                        className="six columns inner-row",
                        style={'width': '70%'},
                        children=[
                            #html.Br(),
                            html.Button("Сохранить", id="btn2"),
                            html.Button("Удалить", id="btn3"),
                            html.Br(),
                            ]
                        ),

                ],
                
            ),

            html.Div(
                id="logs_panel",
                className="six columns",
                children=make_logs_panel(),
                #style={'float': 'right'}
                ),
        ],
    ),

    html.Div(
        className="row",
        id="bottom-row4",
        children=[
            
            html.Div(
                id="islands3",
                className="six columns",

                children=[
                    dcc.Graph(
                        id='all_anomalies_on_groups3',
                        figure={
                            'data':[
                                {'y': ['Об 1', 'Об 2', 'Об 3', 'Об 4', 'Об 5'],
                                 'x': [1000, 100, 50, 30], 'type': 'bar', 'name': 'Вспомогательный',
                                 'text':['Впуск №4', 'Канал №1', 'Канал №2', 'Прокладка выпуска', 'Протечка'],
                                 'textposition':'outside',
                                 'marker' :dict(
                                        color='rgb(100, 83, 109)'
                                        ),
                                'orientation':'h',
                                },
                                
                            ],
                            
                            'layout': dict(
                                xaxis={'title': 'Время'},
                                
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                showlegend=False,
                                
                                hovermode='closest'
                                )
                        }
                    )
                ],
                
            ),

            html.Div(
                id="logs_panel2",
                className="six columns",
                children=[],
                ),
        ],
    ),

    html.Div(
        className="row",
        id="bottom-row5",
        children=[
            
            html.Div(
                id="islands4",
                className="six columns",

                children=[
                    dcc.Graph(
                        id='all_anomalies_on_groups4',
                        figure={
                            'data':[
                                {'x': ['июнь', 'июль', 'август', 'сентябрь', 'октябрь'],
                                 'y': [7800, 7800, 7000, 7500, 4000], 'type': 'bar', 'name': 'Главный',
                                 'marker' :dict(
                                        color='rgb(55, 83, 109)'
                                        )},
                                {'x': ['июнь', 'июль', 'август', 'сентябрь', 'октябрь'],
                                 'y': [6000, 6800, 7000, 7500, 3000], 'type': 'bar', 'name': 'Вспомогательный',
                                 'marker' :dict(
                                        color='rgb(100, 83, 109)'
                                        )},
                            ],
                            'layout': dict(
                                xaxis={'title': 'Время'},
                                
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                showlegend=False,
                                hovermode='closest'
                                )
                        }
                    )
                ],
                
            ),

            html.Div(
                id="logs_panel3",
                className="six columns",
                children=[],
                ),
        ],
    ),

    ],
)

app.layout = layout

if __name__ == '__main__':
    app.enable_dev_tools(debug=True, dev_tools_props_check=False)
    app.run_server(debug=True,host=os.getenv("HOST", "localhost"), port=os.getenv("PORT", "9091"),   
                )
