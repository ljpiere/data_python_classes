#!/usr/bin/python
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html

import plotly.graph_objs as go

import pandas as pd

#----------------------------------------------------------------------------------
#
# Exploremos dash
#
#----------------------------------------------------------------------------------
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(
    __name__, external_stylesheets=external_stylesheets, compress=False
)

app.layout = html.Div(
    children=[
        # hacer un encabezado con una etiqueta HTML
        html.H1(children='Dashboard de la muestra, ¡solo juega!'),
        dcc.Graph(
            id='Sample'
        ),
    ]
)

# lógica del dashboard, no cambies las líneas a continuación
if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)