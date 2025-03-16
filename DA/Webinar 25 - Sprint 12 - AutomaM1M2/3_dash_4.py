#!/usr/bin/python
# -*- codificación: utf-8 -*-

import os
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import plotly.graph_objs as go
from datetime import datetime
import pandas as pd

# Construir la ruta absoluta al CSV (se asume que el archivo está en el mismo directorio que el script)
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'olympics.csv')

# Cargar los datos desde el CSV
df = pd.read_csv(csv_path)

# Convertir la columna 'Year' a datetime (se asume formato 'YYYY')
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Asegurarse de que las columnas de medallas sean numéricas
medal_columns = ['Gold', 'Silver', 'Bronze']
for col in medal_columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Agrupar datos por año sumando las medallas (en caso de haber varias filas por año)
olympics_grouped = (
    df.groupby('Year')
    .agg({'Gold': 'sum', 'Silver': 'sum', 'Bronze': 'sum'})
    .reset_index()
)

# Definición del layout
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, compress=False)

app.layout = html.Div(
    children=[
        # Encabezado
        html.H1(children='Medallas Olímpicas por Año'),
        # Selector de rango de fechas
        html.Label('Time range:'),
        dcc.DatePickerRange(
            start_date=olympics_grouped['Year'].dt.date.min(),
            end_date=datetime(2016, 1, 1).strftime('%Y-%m-%d'),
            initial_visible_month=datetime(2016, 1, 1).strftime('%Y-%m-%d'),
            display_format='YYYY-MM-DD',
            id='dt_selector',
        ),
        # Selector de modo de visualización: valores absolutos o relativos
        html.Label('Display mode:'),
        dcc.RadioItems(
            options=[
                {'label': 'Absolute values', 'value': 'absolute_values'},
                {
                    'label': '% from the total number of medals',
                    'value': 'relative_values',
                },
            ],
            value='absolute_values',
            id='mode_selector',
        ),
        # Gráfico de medallas por año
        dcc.Graph(id='medals_by_year'),
    ]
)

# Callback para actualizar el gráfico según el rango de fechas y el modo seleccionado
@app.callback(
    Output('medals_by_year', 'figure'),
    [
        Input('dt_selector', 'start_date'),
        Input('dt_selector', 'end_date'),
        Input('mode_selector', 'value'),
    ],
)
def update_figures(start_date, end_date, mode):
    # Convertir parámetros de entrada a objetos datetime
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    
    # Filtrar los datos según el rango de fechas
    filtered_data = olympics_grouped.query(
        'Year >= @start_date and Year <= @end_date'
    ).copy()

    # Si se selecciona el modo relativo, calcular el porcentaje de cada tipo de medalla por año
    if mode == 'relative_values':
        # Calcular el total de medallas por año
        filtered_data['total'] = filtered_data[medal_columns].sum(axis=1)
        # Evitar divisiones por cero
        filtered_data.loc[filtered_data['total'] == 0, 'total'] = 1
        # Calcular el porcentaje para cada tipo
        for col in medal_columns:
            filtered_data[col] = filtered_data[col] / filtered_data['total']
    
    # Crear las trazas (una por cada tipo de medalla)
    data = []
    for col in medal_columns:
        data.append(
            go.Scatter(
                x=filtered_data['Year'],
                y=filtered_data[col],
                mode='lines',
                stackgroup='one',
                name=col,
            )
        )
        
    # Configurar y devolver la figura
    return {
        'data': data,
        'layout': go.Layout(
            xaxis={'title': 'Date and time'},
            yaxis={'title': 'Medals' if mode == 'absolute_values' else 'Fraction of total medals'},
        ),
    }

if __name__ == '__main__':
    app.run_server(host='0.0.0.0', port=3000)
