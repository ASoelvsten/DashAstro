#nImport required libraries
#import json
import os
from random import randint

import chart_studio.plotly as py
from plotly.graph_objs import *

import FPS
import glob
import plotly.express as px
import numpy as np
import flask
import dash
from dash.dependencies import Input, Output
import dash_core_components as dcc
import dash_html_components as html

import dash_bootstrap_components as dbc

# Setup the app
# The template is configured to execute 'server' on 'app.py'
server = flask.Flask(__name__)
server.secret_key = os.environ.get('secret_key', str(randint(0, 1000000)))

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, server=server, external_stylesheets=external_stylesheets)

# Functions to load MESA and GYRE files

def read_history(name):

    # Function to read history files from MESA
    
    dct = {}
    f = open(name)
    for i, line in enumerate(f):
        if i == 5:
            keys = line.split()
            break
    f.close()
    data = np.genfromtxt(name,skip_header=5)
    data= data[~np.isnan(data).any(axis=1)]
    
    for j, key in enumerate(keys):
        dct[key] = data[:,j]

    return dct

def gyre_read(name):
    data = np.genfromtxt(name,skip_header=5)
    # See https://bitbucket.org/rhdtownsend/gyre/wiki/Output%20Files%20(5.0)
    l = data[:,0]
    n = data[:,2]
    v = data[:,4]
    I = data[:,7]

    mask0 = l == 0
    mask1 = l == 1
    mask2 = l == 2

    return l[mask0], n[mask0], v[mask0], I[mask0], l[mask1], n[mask1], v[mask1], I[mask1], l[mask2], n[mask2], v[mask2], I[mask2]

def echelle(name):
    l0, n0, v0, I0, l1, n1, v1, I1, l2, n2, v2, I2 = gyre_read(name)
    mdnu = np.mean(np.diff(v0))
    x0 = np.mod(v0,mdnu)
    x1 = np.mod(v1,mdnu)
    x2 = np.mod(v2,mdnu)
    return mdnu, x0, v0, x1, v1, x2, v2

Mass = []
FeH = []
where = []
with open("./OUTREACH_GRID/overview.txt") as f:
    lines = f.readlines()
    for line in lines:
        words = line.split()
        Mass.extend([float(words[0])])
        FeH.extend([float(words[1])])
        where.extend([words[2]])

Mass = np.asarray(Mass)
FeH = np.asarray(FeH)
where = np.asarray(where)

ii = 1
jj = 7
df = read_history(where[ii]+"LOGS/history.data")

# Put your Dash code here

app.layout = html.Div([
    html.Div([
        html.Div([
            html.H1(children="Stellar Evolution Models",className = 'eight columns', style={'font-family':"Arial", 'color': "#8B0000", 'fontSize': 72}),
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

    dcc.Markdown(''' This python app gives you the opportunity to explore the properties of stellar models. Below, you can choose between a set of global stellar parameters and see the predicted evolution depicted in an [Hertzsprung-Russel diagram (HR)](https://en.wikipedia.org/wiki/Hertzsprung%E2%80%93Russell_diagram). The other plots show different seismic and structural properties at a certain snapshot of the stellar evolution. Hover above the HR diagram with you cursor to see how these properties change along the stellar evolution tracks.
                 ''',
    className="twelve columns", style={
                'backgroundColor': 'rgb(250, 250, 250)'},
    ),

    html.Div([
        html.Div([
            html.Div(["Mass",dcc.Slider(
                    id='mass-slider',
                    min=min(Mass),
                    max=max(Mass),
                    value=Mass[ii],
                    marks={str(m): str(m) for m in set(Mass)},
                    step=None
                ),]),
            ],
        style={'width': '49%', 'height':'100%' , 'display': 'inline-block'}),
        html.Div([
             html.Div(["Metallicity" ,dcc.Slider(
                    id='FeH-slider',
                    min=min(FeH),
                    max=max(FeH),
                    value=FeH[ii],
                    marks={str(f): str(round(f,2)) for f in set(FeH)},
                    step=None,)])],
                style={'width': '49%','float': 'right', 'display': 'inline-block'}),],
                style={
                'borderBottom': 'thin lightgrey solid',
                'backgroundColor': 'rgb(250, 250, 250)',
                'padding': '10px 5px'
    }),

    html.Div([
        dcc.Graph(id='HRD', hoverData={'points': [{'x':"5772","y":"1"}]}, ),
        dcc.RadioItems(
                id='evostyle',
                options=[{'label': i, 'value': i} for i in ['HR Diagram', 'Kiel Diagram']],
                value='HR Diagram',
                labelStyle={'display': 'inline-block'}
        ),
    ],style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='Echelle'),
        dcc.RadioItems(
                id='freq-type',
                options=[{'label': i, 'value': i} for i in ['Echelle Diagram', 'Power Spectrum']],
                value='Echelle Diagram',
                labelStyle={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block', 'width': '49%'}),

    html.Div([
    dcc.Graph(id='Kippenhahn'),
    dcc.Checklist(id='Kcheck',
        options=[
            {'label': 'Convection (magenta)', 'value': 'CONV'},
            {'label': 'Nuclear burning (blue)', 'value': 'BURN'},
        ],
        value=['CONV', 'BURN'],
        labelStyle={'display': 'inline-block'}
    ),
    ],style={'width': '49%', 'display': 'inline-block', 'padding': '0 20'}),
    html.Div([
        dcc.Graph(id='Composition'),
        dcc.Checklist(id='surface_comp',
        options=[
            {'label': 'H1', 'value': 'h1'},
            {'label': 'He4', 'value': 'he4'},
            {'label': 'Li7', 'value': 'li7'},
            {'label': 'C12', 'value': 'c12'},
            {'label': 'N14', 'value': 'n14'},
            {'label': 'O16', 'value': 'o16'},
            {'label': 'C/N', 'value': 'C/N'},
        ],
        value=['h1', 'he4'],
        labelStyle={'display': 'inline-block'}
        ),
    ], style={'display': 'inline-block', 'width': '49%'}),


    html.Br(),
    html.Br(),
    html.Div(id='my-output'),

    html.Br(),

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

@app.callback(
    Output('my-output', 'children'),
    [Input('mass-slider', 'value'),
     Input('FeH-slider', 'value'),
     Input('HRD', 'hoverData'),
    ])
def update_output_div(selected_mass,selected_FeH,cdata):
    return 'You chose a star with a mass of {} solar masses, metallicity of {} dex and age of {} Myr.'.format(round(selected_mass,2),round(selected_FeH,2),round(df['star_age'][jj]/10**6,0))

@app.callback(
    Output('Composition', 'figure'),
    [Input('mass-slider', 'value'),
     Input('FeH-slider', 'value'),
     Input('HRD', 'hoverData'),
     Input('surface_comp', 'value')
    ])
def updater_comp(selected_mass,selected_FeH,cdata,surface_comp):

    lab = ['h1','he4','li7','c12','n14','o16']

    c = ['indigo','green','orange','olive','cyan','magenta',]

    age = df['star_age']/10**6

    fig = Figure()

    for j, i in enumerate(lab):
        if i in surface_comp:
            sc=df['surface_'+i]
            fig.add_trace(Scatter(x=age, y=sc,
            mode='lines', line_color=c[j],name=i))

    umax = 1

    if 'C/N' in surface_comp:
        sc=df['surface_c12']/df['surface_n14']
        fig.add_trace(Scatter(x=age, y=sc,
            mode='lines', line_color='black',name='C/N'))
        umax = max([1,max(sc)])+0.1

    iage = age[jj]
    
    fig.add_trace(Scatter(x=[iage,iage], y=[0,umax],
                fill=None,
                mode='lines',
                line_color='red',
                name='Age',
                ))

    fig.update_layout(title="Surface composition")

    fig.update_xaxes(title='Age [Myr]',gridwidth=1,range=[min(age),max(age)])

    fig.update_yaxes(title='Fraction',gridwidth=1)

    return fig

@app.callback(
    Output('Kippenhahn', 'figure'),
    [Input('mass-slider', 'value'),
     Input('FeH-slider', 'value'),
     Input('HRD', 'hoverData'),
     Input('Kcheck', 'value')
    ])
def updater_roundstar(selected_mass,selected_FeH,cdata,Kcheck):

    akeys = df.keys()
#'model_number', 'star_age', 'star_mass', 'effective_T', 'photosphere_L', 'photosphere_r', 'luminosity', 'radius', 'gravity', 'center_T', 'center_Rho', 'center_P', 'center_degeneracy', 'center_mu', 'center_ye', 'center_h1', 'center_he4', 'center_c12', 'center_n14', 'center_o16', 'center_ne22', 'mass_conv_core', 'he_core_mass', 'conv_mx1_top', 'conv_mx1_bot', 'conv_mx2_top', 'conv_mx2_bot', 'mx1_top', 'mx1_bot', 'mx2_top', 'mx2_bot', 'conv_mx1_top_r', 'conv_mx1_bot_r', 'conv_mx2_top_r', 'conv_mx2_bot_r', 'mx1_top_r', 'mx1_bot_r', 'mx2_top_r', 'mx2_bot_r', 'epsnuc_M_1', 'epsnuc_M_2', 'epsnuc_M_3', 'epsnuc_M_4', 'epsnuc_M_5', 'epsnuc_M_6', 'epsnuc_M_7', 'epsnuc_M_8', 'log_LH', 'log_LHe', 'log_LZ', 'log_extra_L', 'log_Lneu', 'log_abs_Lgrav', 'pp', 'cno', 'tri_alfa', 'burn_c', 'burn_n', 'burn_o', 'max_eps_h', 'max_eps_h_m', 'max_eps_h_lgR', 'max_eps_he', 'max_eps_he_m', 'max_eps_he_lgR', 'max_eps_z', 'max_eps_z_m', 'max_eps_z_lgR', 'delta_nu', 'delta_Pg', 'nu_max', 'log_L', 'surface_h1', 'surface_h2', 'surface_he3', 'surface_he4', 'surface_li7', 'surface_be7', 'surface_b8', 'surface_c12', 'surface_n14', 'surface_o16', 'surface_ne20', 'surface_mg24', 'burn2_top1', 'burn2_bot1', 'burn2_top2', 'burn2_bot2', 'burn2_top3', 'burn2_bot3', 'burn2_top4', 'burn2_bot4', 'coupling'
    age = df['star_age']/10**6
    bot = []
    for i in akeys:
        if 'bot' in i:
            bot.extend([i])

    fig = Figure()

    B = []
    T = []

    theta = np.linspace(0,np.pi,100)

    x1 = np.cos(theta)*0
    y1 = np.sin(theta)*0
    x2 = np.cos(theta)*1
    y2 = np.sin(theta)*1
    fig.add_trace(Scatter(x=np.concatenate([x2, x1[::-1]]), y=np.concatenate([y2, y1[::-1]]),
    fill='toself', # fill area
    mode='lines', line_color='yellow'))

    theta2 = np.linspace(np.pi,np.pi*2,100)

    x1 = np.cos(theta2)*0
    y1 = np.sin(theta2)*0
    x2 = np.cos(theta2)*1
    y2 = np.sin(theta2)*1
    fig.add_trace(Scatter(x=np.concatenate([x1, x2[::-1]]), y=np.concatenate([y1, y2[::-1]]),
    fill='toself', # fill area
    mode='lines', line_color='yellow'))

    if 'CONV' in Kcheck:
        for i in bot:
            if 'conv' in i and "_r" not in i:
                top = i.replace("bot","top")
                b = df[i][jj]
                t = df[top][jj] 
                x1 = np.cos(theta)*b
                y1 = np.sin(theta)*b
                x2 = np.cos(theta)*t
                y2 = np.sin(theta)*t
                fig.add_trace(Scatter(x=np.concatenate([x2, x1[::-1]]), y=np.concatenate([y2, y1[::-1]]),
                fill='toself', # fill area
                mode='lines', line_color='magenta'))

                x1 = np.cos(theta2)*b
                y1 = np.sin(theta2)*b
                x2 = np.cos(theta2)*t
                y2 = np.sin(theta2)*t
                fig.add_trace(Scatter(x=np.concatenate([x1, x2[::-1]]), y=np.concatenate([y1, y2[::-1]]),
                fill='toself', # fill area
                mode='lines', line_color='magenta'))


    if 'BURN' in Kcheck:
        for i in bot:
            if 'burn2' in i and "_r" not in i:
                top = i.replace("bot","top")
                b = df[i][jj]
                t = df[top][jj]
                x1 = np.cos(theta)*b
                y1 = np.sin(theta)*b
                x2 = np.cos(theta)*t
                y2 = np.sin(theta)*t
                fig.add_trace(Scatter(x=np.concatenate([x2, x1[::-1]]), y=np.concatenate([y2, y1[::-1]]),
                fill='toself', # fill area
                mode='lines', line_color='indigo'))

                x1 = np.cos(theta2)*b
                y1 = np.sin(theta2)*b
                x2 = np.cos(theta2)*t
                y2 = np.sin(theta2)*t
                fig.add_trace(Scatter(x=np.concatenate([x1, x2[::-1]]), y=np.concatenate([y1, y2[::-1]]),
                fill='toself', # fill area
                mode='lines', line_color='indigo'))

#                fig.add_trace(Scatter(x=np.concatenate([age, age[::-1]]), y=np.concatenate([df[top], df[i][::-1]]),
#                fill='toself', # fill area
#                mode='lines', line_color='orange'))

#    iage = age[jj]
    
#    fig.add_trace(Scatter(x=[iage,iage], y=[0,1],
#                fill=None,
#                mode='lines',
#                line_color='red',
#                ))

#    h1 = df['center_h1']

#    sage = age[[index for index, value in enumerate(h1) if value < 1e-5][0]]

    fig.update_layout(showlegend=False,width=380,height=400)

    fig.update_layout(title="Cross section (structure)")

    fig.update_xaxes(title='Mass [solar masses]',gridwidth=1)

    fig.update_yaxes(title='Mass [solar masses]',gridwidth=1)

#    fig.update_xaxes(type='log')

    return fig

@app.callback(
    Output('HRD', 'figure'),
    [Input('evostyle','value'),
     Input('mass-slider', 'value'),
     Input('FeH-slider', 'value'),
     Input('HRD', 'hoverData'),
    ])
def update_hrd(evostyle,selected_mass,selected_FeH,cdata):
 
    # Find the closest in Euclidean distance
    global ii
    ii = ((Mass-selected_mass)**2+(FeH-selected_FeH)**2).argmin()

    tt = float(cdata["points"][0]['x'])
    ll = float(cdata["points"][0]['y'])

    global df
    df = read_history(where[ii]+"LOGS/history.data")

    L = df["luminosity"]
    Teff = df["effective_T"]
    logg = np.log10(df['gravity'])

    global jj
    if evostyle == 'HR Diagram':
        jj = ((Teff-tt)**2+(L-ll)**2).argmin()
    else:
        jj = ((Teff-tt)**2+(logg-ll)**2).argmin()

    dp = {"select_T":Teff[jj],
          "select_L":L[jj],
          "select_g":logg[jj],}

    if evostyle == "HR Diagram":

        fig = px.line(df, x="effective_T", y="luminosity")
 
        fig.add_scatter(x=[dp['select_T']], y=[dp['select_L']],marker=dict(size=10,color="red"))

        fig.update_yaxes(title='Luminosity [solar luminosity]', mirror=True, type = 'log')

    else:

        dff = {'T':df['effective_T'],
               'g':np.log10(df['gravity']),}

        fig = px.line(dff, x="T", y="g")
        fig.update_yaxes(title='logg',range=[max(logg)+0.1,min(logg)-0.1])

        fig.add_scatter(x=[dp['select_T']], y=[dp['select_g']],marker=dict(size=10,color="red"))

    fig.update_layout(showlegend=False) 

    fig.update_layout(title="Stellar evolution path")

    fig.update_xaxes(title='Effective temperature [K]',range=[max(df['effective_T']),min(df['effective_T'])] )

    return fig

@app.callback(
    Output('Echelle', 'figure'),
    [Input('freq-type','value'),
     Input('mass-slider', 'value'),
     Input('FeH-slider', 'value'),
     Input('HRD', 'hoverData'),
    ])
def update_echelle(freqstyle,selected_mass,selected_FeH,cdata):

    # Associate numbers with profile numbers

    Profiles = glob.glob(where[ii]+"LOGS/*.index")
    Profiles = np.genfromtxt(Profiles[0],skip_header=1)
    nrpr = Profiles[:,0]
    prof = Profiles[:,2]

    # Which profiles have frequencies (all that are all in the history file)

    nr = df["model_number"]

    i_select = abs(nrpr-nr[jj]).argmin()

    gyre_file = where[ii] + "FREQS/" + where[ii][len("./OUTREACH_GRID/"):-1] + "_n" + str(int(prof[i_select])) + ".profile.FGONG.sgyre_l0"

    if freqstyle == "Power Spectrum":
        v0, a0 = FPS.fakePspec(df['star_mass'][jj],df['photosphere_r'][jj],df['effective_T'][jj],gyre_file)
        dv = {'v0':v0,
              'a0':a0,
              }
        fig2 = px.line(dv,x='v0',y='a0')  
        fig2.update_xaxes(title='Frequency [microHertz]')
        fig2.update_yaxes(title='Amplitude [arb. unit]')

    else:
        mdnu, x0, v0, x1, v1, x2, v2 = echelle(gyre_file)
        dv = {'v0':v0,
              'x0':x0,
              }

        fig2 = px.scatter(dv,x='x0',y='v0')

        fig2.update_xaxes(title='Frequency modulo large frequency separation',range=[0,mdnu])

        fig2.update_yaxes(title='Frequecy [microHertz]')

    fig2.update_layout(title="Oscillation frequencies")

    return fig2

# Run the Dash app
if __name__ == '__main__':
    app.server.run()
