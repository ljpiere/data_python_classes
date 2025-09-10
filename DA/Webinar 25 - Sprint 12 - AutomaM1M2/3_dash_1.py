#!/usr/bin/python
# -*- codificación: utf-8 -*-

import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import math
#------------------------------------------------------------------------------------
# definir los datos a mostrar
x = range(-100, 100, 1)
x = [x / 10 for x in x]
y_sin = [math.sin(x) for x in x]
y_cos = [math.cos(x) for x in x]
#------------------------------------------------------------------------------------
data = [
    go.Scatter(
        x=pd.Series(x), y=pd.Series(y_sin), mode='lines', name='sin(x)'
    ),
    go.Scatter(
        x=pd.Series(x), y=pd.Series(y_cos), mode='lines', name='cos(x)'
    ),
]
#------------------------------------------------------------------------------------
# definir el diseño
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    children=[
        # html
        html.H1(children='Funciones trigonométricas'),
        dcc.Graph(
            figure={
                'data': data,
                'layout': go.Layout(
                    xaxis={'title': 'x'}, yaxis={'title': 'y'}
                ),
            },
            id='trig_func',
        ),
    ]
)
#------------------------------------------------------------------------------------
# lógica del dashboard
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)