import dash
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
from sqlalchemy import create_engine

# conexión a PostgreSQL
engine = create_engine("postgresql://airflow:airflow@localhost:5432/airflow")

# leer datos desde la tabla
df = pd.read_sql("SELECT x_value, y_sin, y_cos FROM trig_functions ORDER BY x_value", engine)

# definir datos
data = [
    go.Scatter(
        x=df['x_value'], y=df['y_sin'], mode='lines', name='sin(x)'
    ),
    go.Scatter(
        x=df['x_value'], y=df['y_cos'], mode='lines', name='cos(x)'
    ),
]

# diseño
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
app.layout = html.Div(
    children=[
        html.H1(children='Funciones trigonométricas desde PostgreSQL'),
        dcc.Graph(
            figure={
                'data': data,
                'layout': go.Layout(xaxis={'title': 'x'}, yaxis={'title': 'y'}),
            },
            id='trig_func',
        ),
    ]
)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
