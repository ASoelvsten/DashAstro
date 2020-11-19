#Import required libraries
#import json
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

# Add css to be able to change styling
#app.css.append_css({'external_url': 'https://codepen.io/amyoshino/pen/jzXypZ.css'})

# We load the APOKASC DATA
filename = "./table5.dat"

KIC = []
Teff = []
M = []
FeH = []
R = []
logg = []
evo = []
e_Teff = []
e_FeH = []
AFe = []
e_AFe = []
Numax = []
e_Numax = []
Dnu = []
e_Dnu = []
e_logg = []
Rho = []
e_Rho = []
e_R = []
e_M = [] 

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
            e_Teff.extend([float(line[35:40])])
            e_FeH.extend([float(line[48:53])])
            AFe.extend([float(line[54:60])])
            e_AFe.extend([float(line[61:66])])
            Numax.extend([float(line[67:74])])
            e_Numax.extend([float(line[75:80])])
            Dnu.extend([float(line[81:87])])
            e_Dnu.extend([float(line[88:93])])
            e_logg.extend([float(line[160:165])])
            Rho.extend([float(line[172:181])])
            e_Rho.extend([float(line[182:187])])
            e_R.extend([float(line[142:147])])
            e_M.extend([float(line[123:128])])

KIC = np.asarray(KIC)
Teff = np.asarray(Teff)
M = np.asarray(M)
FeH = np.asarray(FeH)
R = np.asarray(R)
logg = np.asarray(logg)
evo = np.asarray(evo)
e_Teff = np.asarray(e_Teff)
e_FeH = np.asarray(e_FeH)
AFe   = np.asarray(AFe)
e_AFe = np.asarray(e_AFe)
Numax = np.asarray(Numax)
e_Numax = np.asarray(e_Numax)
Dnu = np.asarray(Dnu)
e_Dnu = np.asarray(e_Dnu)
e_logg = np.asarray(e_logg)
Rho = np.asarray(Rho)
e_Rho = np.asarray(e_Rho)
e_R = np.asarray(e_R)
e_M = np.asarray(e_M)

df = {"Mass":M,
      "Effective temperature [K]":Teff,
      "Radius":R,
      "Metallicity":FeH,
      "Surface gravity":logg,
      "Mean Density":Rho,
      "Alpha enrichment":AFe,
      "Frequency of maximum power [µHz]":Numax,
      "Large frequency separation [µHz]":Dnu,
}

available_indicators = df.keys()

Mdata = pd.read_csv('Miglioetal_CATALOGUE.csv',sep=r'\s*,\s*',engine='python')

dMi = {'Mass':Mdata.mass,
       'Mass error':Mdata.mass_err18+Mdata.mass_err84-2*Mdata.mass,
       'Age [Gyr]':Mdata.age,
       'Age error [Gyr]':Mdata.age_err18+Mdata.age_err84-2*Mdata.age,
       'Radius':Mdata.rad,
       'Radius error': Mdata.rad_err18+Mdata.rad_err84-2*Mdata.rad,
       'Metallicity':Mdata.FE_H_APOGEE,
       'Alpha enrichment':Mdata.ALPHA_M_APOGEE,
       'Distance':Mdata.dist,
}

# Put your Dash code here

evstate = ['All','RGB','RC']

#============================================================================================================================
# MAIN
#============================================================================================================================

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1(children="APOKASC-2 Data",className = 'eight columns', style={'font-family':"Arial", 'color': "#8B0000", 'fontSize': 72}),
            html.Div([
                html.A([html.Img(src="https://www.asterochronometry.eu/images/Asterochronometry_full.jpg", className = 'four columns',
                     style={'height':'13%',
                            'width':'13%',
                            'float':'right',
                            'position':'relative',
                            'margin-top':10,
                            'margin-left':10,
                            'margin-right':10,
                            'margin-bottom':10
                            },
                )], href="https://www.asterochronometry.eu/",id="astrologo"),
                dbc.Tooltip(" Visit our webside ",target="astrologo",placement="bottom",style={'padding':'5px','color':"white",'background-color':"black"}),        
            ]),


        ], className = "row"),
    ]),

    dcc.Markdown(''' This python app gives you the opportunity to explore the so-called [APOKASC-II](https://arxiv.org/abs/1804.09983) sample. This data set combines asteroseismic data from the _Kepler_ space mission with spectroscopic constraints from [APOGEE](https://www.sdss3.org/surveys/apogee.php). The underlying data is publically available and can be found [here](https://cdsarc.unistra.fr/viz-bin/cat/J/ApJS/239/32). We use such data in our own research in the asterochronometry group. Check us out on our [homepage](https://www.asterochronometry.eu/). Do you have a specific star in mind? Then, just enter the KIC number below to access information about the star. Otherwise, explore the sample using our interactive plot below.
                 ''',
                 className="twelve columns"),

    html.Datalist(
    id='list-suggested-inputs',
    children=[html.Option(value=int(k)) for k in KIC]),

    html.Div(["KIC: ",
              dcc.Input(id='my-input', list='list-suggested-inputs', value='1027110', type='text')]),

    html.Br(),
    html.Div(id='my-output'),
    html.Br(),
    dcc.Markdown(''' Here, we distiguish between red giant branch ([RGB](https://en.wikipedia.org/wiki/Red-giant_branch)) stars, red clump ([RC](https://en.wikipedia.org/wiki/Red_clump)) stars, and stars, whose evolutionary stage is ambigious (AMB).  
    '''
    ),
    html.Br(),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id='xaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Effective temperature [K]'
            ),
            dcc.RadioItems(
                id='xaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],
        style={'width': '48%', 'display': 'inline-block'}),

        html.Div([
            dcc.Dropdown(
                id='yaxis-column',
                options=[{'label': i, 'value': i} for i in available_indicators],
                value='Surface gravity'
            ),
            dcc.RadioItems(
                id='yaxis-type',
                options=[{'label': i, 'value': i} for i in ['Linear', 'Log']],
                value='Linear',
                labelStyle={'display': 'inline-block'}
            )
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    # First graph
    dcc.Graph(id='graph1'),

    html.Label([
        "Evolutionary stage",
        dcc.Dropdown(
            id='Evolutionary state', clearable=False,
            value='RGB', options=[
                {'label': c, 'value': c}
                for c in ["RC","RGB","AMB"]

            ])
    ]),

    html.Br(),

    html.P("Click on the graph to select any star in a sample. The global properties of the selected target will be shown in the table below. To select several targets, hold down the shift buttom while clicking."),

    html.Div(id='tab1'),

    # ----  MIGLIO et al. 2020 ----

    html.H2(children="Miglio et al. (2020)"),

    dcc.Markdown(''' To exemplify how such data comes into play in our work, we include results from a recent paper by [Miglio et al. (2020)](https://ui.adsabs.harvard.edu/abs/2020arXiv200414806M/abstract). You can explore the results below.
                 ''',
                 className="twelve columns"),

    html.Br(),

    html.Label([
        "Evolutionary stage",
         dcc.Dropdown(
            id='evoMig',
            options=[{'label': i, 'value': i} for i in evstate],
            value='All'
        ),
    ]),

    html.Br(),

    html.Div([
        dcc.Dropdown(
            id='Mdrop',
            options=[{'label': i, 'value': i} for i in dMi.keys()],
            value="Age [Gyr]",
            ),
        dcc.Graph(id='graph2'),
    ], style={'width': '48%', 'display': 'inline-block'}),
    html.Div([
        dcc.Dropdown(
            id='Mdrop2',
            options=[{'label': i, 'value': i} for i in dMi.keys()],
            value="Age error [Gyr]",
            ),
        dcc.Graph(id='graph3'),
    ], style={'width': '48%', 'float': 'right', 'display': 'inline-block'}),

    html.Br(),

    html.Div([
        html.Div([
            dcc.Dropdown(
                id='xaxis-column2',
                options=[{'label': i, 'value': i} for i in dMi.keys()],
                value='Age [Gyr]'
            ),
        ],
        style={'width': '48%', 'display': 'inline-block'}),
        html.Div([
            dcc.Dropdown(
                id='yaxis-column2',
                options=[{'label': i, 'value': i} for i in dMi.keys()],
                value='Mass'
            ),
        ],style={'width': '48%', 'float': 'right', 'display': 'inline-block'})
    ]),

    # First graph
    dcc.Graph(id='graph4'),

    # ---- FOOTER ----

    html.Br(),

    dcc.Markdown("Visit our GitHub folder by clicking on the logo below or check out some of our other apps: [Visualization of stellar evolution](https://evostar.herokuapp.com/)."),

    html.Div([
        html.Div([
            html.A([html.Img(src="https://github.githubassets.com/images/modules/logos_page/GitHub-Mark.png", className = 'four columns',
                     style={'height':'5%',
                            'width':'5%',
                            'float':'left',
                            'position':'relative',
                            'margin-top':10,
                            'margin-left':10,
                            'margin-right':10,
                            'margin-bottom':10
                            },
            )], href="https://github.com/ASoelvsten/DashAstro", id="ghlogo"),
            dbc.Tooltip(" Visit our GitHub folder ",target="ghlogo",placement="top",style={'padding':'5px','color':"white",'background-color':"black"}),]),
        html.Div([
            html.A([html.Img(src="https://upload.wikimedia.org/wikipedia/commons/thumb/3/37/Plotly-logo-01-square.png/1200px-Plotly-logo-01-square.png", className = 'eight columns',
                     style={'height':'8%',
                            'width':'8%',
                            'float':'right',
                            'position':'relative',
                            'margin-top':10,
                            'margin-left':10,
                            'margin-right':10,
                            'margin-bottom':10
                            },
            )], href="https://dash.plotly.com/", id="dashlogo"),
            dbc.Tooltip(" Make your own dashboards ",target="dashlogo",placement="top",style={'padding':'5px','color':"white",'background-color':"black"}),]),
    ], className = "row"),

])

#============================================================================================================================
# CALLBACKS
#============================================================================================================================

# Define callback to update graph1
@app.callback(
    Output('graph4', 'figure'),
    [Input("xaxis-column2", "value"),
     Input("yaxis-column2", "value"),
     Input("evoMig", "value")]
)
def update_figure2(xaxis_column_name, yaxis_column_name,evostat):

    if evostat == "All":
        mask = Mdata.evstate >= -1.
    if evostat == "RC":
        mask = Mdata.evstate == 2
    if evostat == "RGB":
        mask = Mdata.evstate == 1

    KICM = Mdata.KIC_ID[mask]

    dMie = {
       'Mass':Mdata.mass[mask],
       'Mass error':Mdata.mass_err18[mask]+Mdata.mass_err84[mask]-2*Mdata.mass[mask],
       'Age [Gyr]':Mdata.age[mask],
       'Age error [Gyr]':Mdata.age_err18[mask]+Mdata.age_err84[mask]-2*Mdata.age[mask],
       'Radius':Mdata.rad[mask],
       'Radius error': Mdata.rad_err18[mask]+Mdata.rad_err84[mask]-2*Mdata.rad[mask],
       'Metallicity':Mdata.FE_H_APOGEE[mask],
       'Alpha enrichment':Mdata.ALPHA_M_APOGEE[mask],
       'Distance':Mdata.dist[mask],
    }

    for i in dMie.keys():
        if i != xaxis_column_name and i != yaxis_column_name and ("error" not in i):
            c = i
            break

    print(c)

    fig = px.scatter(dMie, x = xaxis_column_name,
                     y = yaxis_column_name,
                     color = c,
                     hover_name=KICM)

    return fig

@app.callback(
    Output('graph3', 'figure'),
    [Input("Mdrop2", "value"),
     Input("evoMig", "value")]
)
def update_figure2(who,evostat):

    if evostat == "All":
        mask = Mdata.evstate >= -1.
    if evostat == "RC":
        mask = Mdata.evstate == 2
    if evostat == "RGB":
        mask = Mdata.evstate == 1

    dMie = {'Mass':Mdata.mass[mask],
       'Mass error':Mdata.mass_err18[mask]+Mdata.mass_err84[mask]-2*Mdata.mass[mask],
       'Age [Gyr]':Mdata.age[mask],
       'Age error [Gyr]':Mdata.age_err18[mask]+Mdata.age_err84[mask]-2*Mdata.age[mask],
       'Radius':Mdata.rad[mask],
       'Radius error': Mdata.rad_err18[mask]+Mdata.rad_err84[mask]-2*Mdata.rad[mask],
       'Metallicity':Mdata.FE_H_APOGEE[mask],
       'Alpha enrichment':Mdata.ALPHA_M_APOGEE[mask],
       'Distance':Mdata.dist[mask],
    }

    fig = px.histogram(dMie, x=who,opacity=0.8,color_discrete_sequence=['indigo'])

    return fig

@app.callback(
    Output('graph2', 'figure'),
    [Input("Mdrop", "value"),
     Input("evoMig", "value")]
)
def update_figure2(who,evostat):

    if evostat == "All":
        mask = Mdata.evstate >= -1.
    if evostat == "RC":
        mask = Mdata.evstate == 2
    if evostat == "RGB":
        mask = Mdata.evstate == 1

    dMie = {'Mass':Mdata.mass[mask],
       'Mass error':Mdata.mass_err18[mask]+Mdata.mass_err84[mask]-2*Mdata.mass[mask],
       'Age [Gyr]':Mdata.age[mask],
       'Age error [Gyr]':Mdata.age_err18[mask]+Mdata.age_err84[mask]-2*Mdata.age[mask],
       'Radius':Mdata.rad[mask],
       'Radius error': Mdata.rad_err18[mask]+Mdata.rad_err84[mask]-2*Mdata.rad[mask],
       'Metallicity':Mdata.FE_H_APOGEE[mask],
       'Alpha enrichment':Mdata.ALPHA_M_APOGEE[mask],
       'Distance':Mdata.dist[mask],
    }

    fig = px.histogram(dMie, x=who,opacity=0.8,color_discrete_sequence=['indianred'])

    return fig

@app.callback(
    Output(component_id='my-output', component_property='children'),
    [Input(component_id='my-input', component_property='value')]
)
def update_output_div(input_value):
    KIC_write = int(input_value)
    if KIC_write in KIC:
        mask = KIC == KIC_write
        return 'KIC {} '.format(input_value)+'is an '+evo[mask][0]+' star. It has a mass of {} solar masses'.format(float(M[mask])) + ' and a radius of {} solar radii. '.format(float(R[mask])) +'The metallicity, effective temperature and surface gravity are {} dex, {} K, {} dex, respectively.'.format(float(FeH[mask]),float(Teff[mask]),float(logg[mask])) 
    else:
        return "Please, select a KIC-number in the APOKASC-2 sample."

@app.callback(
    [Output('tab1', 'children'),
     ],
    [Input('graph1', 'selectedData')])
def callback_a(selectData):
    KK = []
    Tc =  []
    Mc = []
    FeHc = []
    Rc = []
    loggc = []
    evoc = []
    e_Teffc = []
    e_FeHc = []
    AFec = []
    e_AFec = []
    Numaxc = []
    e_Numaxc = []
    Dnuc    = []
    e_Dnuc  = []
    e_loggc = []
    Rhoc    = []
    e_Rhoc  = []
    e_Rc = []
    e_Mc = []

    for i, j in enumerate(selectData["points"]):
        KIC_click = selectData["points"][i]["hovertext"]
        mask = KIC == KIC_click
        KK.extend([int(KIC_click)])
        Tc.extend([float(Teff[mask])])
        Mc.extend([float(M[mask])])
        FeHc.extend([float(FeH[mask])])
        Rc.extend([float(R[mask])])
        loggc.extend([float(logg[mask])])
        evoc.extend([evo[mask]])
        e_Teffc.extend([float(e_Teff[mask])])
        e_FeHc.extend([float(e_FeH[mask])])
        AFec.extend([float(AFe[mask])])
        e_AFec.extend([float(e_AFe[mask])])
        Numaxc.extend([float(Numax[mask])])
        e_Numaxc.extend([float(e_Numax[mask])])
        Dnuc.extend([float(Dnu[mask])])
        e_Dnuc.extend([float(e_Dnu[mask])])
        e_loggc.extend([float(e_logg[mask])])
        Rhoc.extend([float(Rho[mask])])
        e_Rhoc.extend([float(e_Rho[mask])])
        e_Rc.extend([float(e_R[mask])])
        e_Mc.extend([float(e_M[mask])])

    tab = html.Table([
       html.Thead(
               html.Tr(
               children=[html.Th("KIC") , html.Th("Mass"), html.Th("Radius"), html.Th("Metallicity"), html.Th("Effective temperature"), html.Th("Alpha enrichment"), html.Th("Stage")],
           ),
       ),
       html.Tbody([
           html.Tr([html.Td(j), html.Td(str(Mc[i])+"±"+str(e_Mc[i])), html.Td(str(Rc[i])+"±"+str(e_Rc[i])), html.Td(str(FeHc[i])+"±"+str(e_FeHc[i])), html.Td(str(Tc[i])+"±"+str(e_Teffc[i])), html.Td(str(AFec[i])+"±"+str(e_AFec[i])), html.Td(evoc[i])]) for i, j  in enumerate(KK)]),

    ], style = {'font-family':"Comic Sans MS"}  ),

    return tab

# Define callback to update graph1
@app.callback(
    Output('graph1', 'figure'),
    [Input('xaxis-column', 'value'),
     Input('yaxis-column', 'value'),
     Input('xaxis-type', 'value'),
     Input('yaxis-type', 'value'),
     Input("Evolutionary state", "value")]
)

def update_figure(xaxis_column_name, yaxis_column_name,
                 xaxis_type, yaxis_type,evostate):
     
    mask = evo == evostate

    dff = {
          "Mass":M[mask],
          "Effective temperature [K]":Teff[mask],
          "Radius":R[mask],
          "Metallicity":FeH[mask],
          "Surface gravity":logg[mask],
          "Mean Density":Rho[mask],
          "Alpha enrichment":AFe[mask],
          "Frequency of maximum power [µHz]":Numax[mask],
          "Large frequency separation [µHz]":Dnu[mask],
    }

    sizes = ["Radius","Mass","Surface gravity"]

    for i in sizes:
        if i != xaxis_column_name and i != yaxis_column_name:
            s = i

    for i in available_indicators:
        if i != xaxis_column_name and i != yaxis_column_name and i !=s:
            c = i
            break

    fig = px.scatter(dff, x = xaxis_column_name,
                     y = yaxis_column_name,
                     color = c,size=s,
                     hover_name=KIC[mask])

    fig.update_xaxes(title=xaxis_column_name, 
                     type='linear' if xaxis_type == 'Linear' else 'log') 

    fig.update_yaxes(title=yaxis_column_name, 
                     type='linear' if yaxis_type == 'Linear' else 'log') 

    fig.update_layout(clickmode='event+select')
   
    return fig

# Run the Dash app
if __name__ == '__main__':
    app.server.run()
