# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from datetime import datetime as dt
import pandas as pd
import numpy as np
from math import trunc,sqrt,ceil

import dash_daq as daq

df = pd.read_csv('https://gist.githubusercontent.com/chriddyp/5d1ea79569ed194d432e56108a04d188/raw/a9f9e8076b837d541398e999dcbac2b2826a81f8/gdp-life-exp-2007.csv')

df1 = pd.read_excel('./ActualDataFeb2020.xlsx')
df2 = pd.read_csv('AutoDataset.csv')
df2.columns = [x[1:] for x in df2.columns]
df2 = df2.dropna()
cols_df2 = ['wheel-base', 'engine-size', 'horsepower']

app = dash.Dash(__name__)

colors_graphs = {
    'background': '#e6e8f2',
    'base1': '#afb0b2',
    'base2': '#203256',
    'base3': '#348a7d',
    'base4': '#c14a5e',
}


def make_ident(k,i,j, size=15):
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
            className="six columns",
            style={'width': '20%'},
            children=[build_graph_title('Island'+str(land))]+[
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


def build_graph_title(title):
    return html.P(className="graph-title", children=title)


app.layout = html.Div(
    style={'background': '#f2f5fa'},
    id="bottom-row",
    children=[
    html.Div(
        className="row",

        children=[

            html.Div(
                id="dropouts-container",
                className="six columns",

                children=[
                    html.Div(
                        id="dropout1",
                        className="six columns",
                        style={'width': '30%'},
                        children=[
                            html.Div(children='''
                                    Период
                                '''),
                            dcc.DatePickerRange(
                                id='date-picker-range',
                                end_date=dt.now(),
                                start_date_placeholder_text='Select a date!',
                            ),
                        ],
                    ),

                    html.Div(
                        id="dropout2",
                        className="six columns",
                        style={'width': '30%'},
                        children=[
                            html.Div(children='''
                                Опция1
                            '''),
                            dcc.Dropdown(
                                options=[
                                    {'label': 'New York City', 'value': 'NYC'},
                                    {'label': 'Montréal', 'value': 'MTL'},
                                    {'label': 'San Francisco', 'value': 'SF'}
                                ],
                                multi=True,
                                value="MTL",
                            ),
                        ],
                    ),

                    html.Div(
                        id="dropout3",
                        className="six columns",
                        style={'width': '30%'},
                        children=[
                                html.Div(children='''
                                    Опция2
                                '''),
                            dcc.Dropdown(
                                options=[
                                    {'label': 'New York City', 'value': 'NYC'},
                                    {'label': 'Montréal', 'value': 'MTL'},
                                    {'label': 'San Francisco', 'value': 'SF'}
                                ],
                                multi=True,
                                value="MTL",
                            ),

                        ],
                    ),
                ],
            ),

            html.Div(
                id="info-container",
                className="six columns",

                children=[
                    html.Div(

                        id="info1",
                        className="six columns",
                        style={'width': '30%'},
                        children=[

                            html.Div(children='''
                                    Общая нагрузка
                                '''),
                            daq.GraduatedBar(
                                id='my-daq-graduatedbar',
                                value=4
                            )
                        ],
                    ),

                    html.Div(
                        id="info2",
                        className="six columns",
                        style={'width': '10%',
                               },
                        children=[
                            html.Div(children='''
                                    Температура
                                '''),
                            build_graph_title('13 C'),

                        ],
                    ),

                    html.Div(
                        id="well-production-container",
                        className="six columns",
                        style={'width': '10%',
                               },
                        children=[
                                html.Div(children='''
                                    Влажность
                                '''),
                                build_graph_title('20 %'),
                            ],
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
                        id='life-exp-vs-gdp1212',
                        figure={
                            'data': [
                                dict(
                                    x=df[df['continent'] == i]['gdp per capita'],
                                    y=(va+1)*np.ones((len(df[df['continent'] == i]['gdp per capita']),)),
                                    text='logs '+str(va)+' land '+df[df['continent'] == i]['country'],
                                    mode='markers',
                                    opacity=0.7,
                                    marker={
                                        'size': 15,
                                        'line': {'width': 0.5, 'color': 'white'}
                                    },
                                    name=i
                                ) for va, i in enumerate(df.continent.unique())
                            ],
                            'layout': dict(
                                xaxis={'title': 'GDP Per Capita'},
                                yaxis={'title': 'Life Expectancy'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
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
                        id='life-exp-vs-gdp1',
                        figure={
                            'data': [
                                dict(
                                    x=df2.index,
                                    y=df2.loc[:, i],
                                    text=df2.loc[:, i].name,
                                    mode='Scatter',

                                    name=i
                                ) for i in cols_df2
                            ],
                            'layout': dict(
                                xaxis={'title': 'GDP Per Capita'},
                                yaxis={'title': 'Life Expectancy'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest'
                            )
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
                id="islands-ind",
                className="six columns",
                children=[


                    make_island(1,10),
                    make_island(2,7),
                    make_island(3,18),
                    make_island(4,5),

                ],

            ),




            html.Div(
                id="Capacity1212",
                className="six columns",

                children=[
                    dcc.Graph(
                        id='life-exp-vs-gdp221',
                        figure={
                            'data': [
                                dict(
                                    x=df2.index,
                                    y=df2.loc[:, cols_df2[0]],
                                    text=df2.loc[:, cols_df2[0]].name,
                                    type='Scatter',

                                    name=cols_df2[0]
                                ),

                                dict(
                                    x=df2.index,
                                    y=-df2.loc[:, cols_df2[1]],
                                    text=df2.loc[:, cols_df2[1]].name,
                                    type='bar',

                                    name=cols_df2[1]
                                ),

                                dict(
                                    x=df2.index,
                                    y=df2.loc[:, cols_df2[2]],
                                    text=df2.loc[:, cols_df2[2]].name,
                                    mode='lines+markers',
                                    name=cols_df2[2]
                                ),

                            ],
                            'layout': dict(
                                xaxis={'title': 'GDP Per Capita'},
                                yaxis={'title': 'Life Expectancy'},
                                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                                legend={'x': 0, 'y': 1},
                                hovermode='closest'
                            )
                        }
                    )
                    ],
            ),
        ],
    ),

    html.Div(
        className="row",
        id='asd',
        children=[

            html.Div(
                id="logs-container",
                className="six columns",

                children=[
                    build_graph_title('Журнал ошибок'),
                    html.Div(
                        id='log1',
                        children=[
                            html.H6(children=[
                                '''21.12.2012''',
                                html.Br(),
                                '''Cбой системы, конец света не произошел'''
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id='log2',
                        children=[
                            html.H6(children=[
                                '''26.12.1991''',
                                html.Br(),
                                '''Cбой идеологии, СССР распался'''
                                ]
                            )
                        ]
                    ),
                    html.Div(
                        id='log3',
                        children=[
                            html.H6(children=[
                                '''15.04.2020-15.05.2020''',
                                html.Br(),
                                '''Cбой Фондового рынка, он сдулся'''
                                ]
                            )
                        ]
                    ),
                ],
            ),
            html.Div(
                children=[

                ]
            )
        ],
    ),

    ],
)


if __name__ == '__main__':
    app.run_server(debug=True)

