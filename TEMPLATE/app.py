# Import Libraries, check with requirements.txt

import os
from random import randint

import chart_studio.plotly as py
from plotly.graph_objs import *

import plotly.express as px
import numpy as np
import flask
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html
import pandas as pd

import dash_bootstrap_components as dbc

# Setup the app
# The template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

# Load data

Mdata = pd.read_csv('Miglioetal_CATALOGUE.csv',sep=r'\s*,\s*',engine='python')

df = {'Mass':Mdata.mass,
       'Mass error':abs(Mdata.mass_err18+Mdata.mass_err84-2*Mdata.mass),
       'Age [Gyr]':Mdata.age,
       'Age error [Gyr]':abs(Mdata.age_err18+Mdata.age_err84-2*Mdata.age),
       'Radius':Mdata.rad,
       'Radius error': abs(Mdata.rad_err18+Mdata.rad_err84-2*Mdata.rad),
       'Metallicity':Mdata.FE_H_APOGEE,
       'Alpha enrichment':Mdata.ALPHA_M_APOGEE,
       'Distance':Mdata.dist,
}

#=====================================
# MAIN, our app
#=====================================

# Put your Dash code here

app.layout = html.Div([

# We can use normal Markdown or Html through special command

    html.H1(children="Simple App Example",className = 'eight columns', style={'font-family':"Arial", 'color': "#8B0000", 'fontSize': 72}),

    dcc.Markdown(''' Here, is an example of Markdown text as used in Jupyter Notebooks. So, you can add [links](https://en.wikipedia.org/wiki/Lynx) etc.
                 ''',
                 className="twelve columns"),

    html.Br(),

# Through plotly, we can add INTERACTIVE PLOTS. So, this is where the interesting things happen!

    dcc.Dropdown(
        id='example_drop',
        options=[{'label': i, 'value': i} for i in df.keys()],
        value="Mass",
        ),
    dcc.Graph(id='graph'),

])

#=====================================
# CALLBACKS, make your app interactive
#=====================================

@app.callback(
    Output('graph', 'figure'),
    [Input("example_drop", "value"),
    ]
)
def update_figure2(xaxis_column_name):

    fig = px.scatter(df, x = xaxis_column_name,
                     y = "Age [Gyr]")

    return fig

#=====================================

# Run the Dash app
if __name__ == '__main__':
    app.server.run()
