from Dashboard.index import app

# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd
import numpy as np
from math import trunc,sqrt,ceil
from dash.dependencies import Input, Output, State
from datetime import datetime, timedelta
import os

from dash_extensions import Download
from dash_extensions.snippets import send_bytes
import xlsxwriter

from DBConnector import DataBase

import dash_daq as daq


if __name__ == '__main__':
    app.run_server(debug=True,host=os.getenv("HOST", "localhost"), port=os.getenv("PORT", "9091"), 
                    
                )