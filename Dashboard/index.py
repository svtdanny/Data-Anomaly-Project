
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import dash_bootstrap_components as dbc

from . Navbar import Navbar

# Connect to main app.py file
from . app import app
from . app import server

# Connect to your app pages
from . apps import dashboard

app.layout = html.Div([
    Navbar(),
    dcc.Location(id = 'url', refresh = False),
    html.Div(id = 'page-content')
])

@app.callback(Output('page-content', 'children'),
            [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return dashboard.layout
    else:
        print("MOTHER FUCKER")
        return dashboard.layout

if __name__ == '__main__':
    app.enable_dev_tools(debug=True, dev_tools_props_check=False)
    app.run_server(debug=True,host=os.getenv("HOST", "localhost"), port=os.getenv("PORT", "9091"),   
                )
