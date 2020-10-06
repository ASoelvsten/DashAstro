# Import required libraries
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


# Setup the app
# Make sure not to change this file name or the variable names below,
# the template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

app = dash.Dash(__name__, server=server)

#server = app.server

filename = "./table5.dat"

KIC = []
Teff = []
M = []
FeH = []
R = []
logg = []
evo = []

with open(filename) as f:
    lines = f.readlines()
    for line in lines:
        if "No" not in line:
            KIC.extend([float(line[0:8])])
            Teff.extend([float(line[28:34])])
            M.extend([float(line[117:122])])
            FeH.extend([float(line[41:47])])
            R.extend([float(line[135:141])])
            logg.extend([float(line[154:159])])
            evo.extend([line[94:102].strip()])

KIC = np.asarray(KIC)
Teff = np.asarray(Teff)
M = np.asarray(M)
FeH = np.asarray(FeH)
R = np.asarray(R)
logg = np.asarray(logg)
evo = np.asarray(evo)

df = {"M":M,
      "Teff":Teff,
      "R":R,
      "FeH":FeH,
      "logg":logg
}

# Put your Dash code here

app.layout = html.Div([
    html.H1("APOKASC-2 data"),
    html.P("This is an example in JupyterDash. Here, we combine Python with hmtl."),
    dcc.Graph(id='graph'),
    html.Label([
        "Evolutionary stage",
        dcc.Dropdown(
            id='Evolutionary state', clearable=False,
            value='RGB', options=[
                {'label': c, 'value': c}
                for c in ["RC","RGB","AMB"]
            ])
    ]),
])
# Define callback to update graph
@app.callback(
    Output('graph', 'figure'),
    [Input("Evolutionary state", "value")]
)
def update_figure(evostate):
     
    mask = evo == evostate

    df = {"M":M[mask],
          "Teff":Teff[mask],
          "R":R[mask],
          "FeH":FeH[mask],
          "logg":logg[mask]
    }

    
    Fig = px.scatter(
        df, x="Teff", y="logg", color="M", size="R",
        color_continuous_scale="plasma",
        render_mode="webgl"
    )
    
    Fig.update_layout(
    xaxis_title=r"Effective temperature",
    yaxis_title=r"Surface gravity",
    )
    
    return Fig

# Run the Dash app
if __name__ == '__main__':
    app.server.run()
