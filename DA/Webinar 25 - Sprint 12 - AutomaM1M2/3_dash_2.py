#!/usr/bin/python
# -*- codificación: utf-8 -*-

import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
import pandas as pd
import os
# definir los datos a mostrar
from sqlalchemy import create_engine
#----------------------------------------------------------------------------------
#
# Exploremos dash
#
#----------------------------------------------------------------------------------

# código de muestra para conectarse a la base de datos con SQLite
# engine = create_engine('sqlite:////db/games.db', echo=False)

# obtención de datos en bruto
#query = ''' SELECT * FROM data_raw '''
# games_raw = pd.io.sql.read_sql(query, con=engine)

#----------------------------------------------------------------------------------
# Obtiene la ruta del directorio donde se encuentra el script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Construye la ruta completa al archivo CSV
csv_path = os.path.join(script_dir, 'olympics.csv')
#----------------------------------------------------------------------------------

# Lee el archivo CSV
df = pd.read_csv(csv_path)

# Convertir la columna 'Year' a datetime (ajusta el formato si es necesario)
df['Year'] = pd.to_datetime(df['Year'], format='%Y')

# Asegurarse de que las columnas de medallas sean numéricas
medal_columns = ['Gold', 'Silver', 'Bronze']
for column in medal_columns:
    df[column] = pd.to_numeric(df[column], errors='coerce')

# Agrupar datos por año y sumar las medallas
df_grouped = (
    df.groupby('Year')
    .agg({
        'Gold': 'sum',
        'Silver': 'sum',
        'Bronze': 'sum'
    })
    .reset_index()
)

# Configurar estilos de línea para cada tipo de medalla
line_styles = {
    'Gold': {'color': 'gold', 'width': 4},
    'Silver': {'color': 'silver', 'width': 3, 'dash': 'dash'},
    'Bronze': {'color': 'brown', 'width': 2, 'dash': 'dot'},
}

# Definir los gráficos para cada medalla
data_medals_by_year = []
for column in line_styles.keys():
    data_medals_by_year.append(
        go.Scatter(
            x=df_grouped['Year'],
            y=df_grouped[column],
            mode='lines',
            line=line_styles[column],
            name=column
        )
    )

# Configurar el dashboard
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets, compress=False)
app.layout = html.Div(
    children=[
        html.H1(children='Medallas Olímpicas por Año'),
        dcc.Graph(
            id='medals_by_year',
            figure={
                'data': data_medals_by_year,
                'layout': go.Layout(
                    title='Medallas Olímpicas por Año',
                    xaxis={'title': 'Año'},
                    yaxis={'title': 'Número de Medallas'}
                )
            }
        )
    ]
)

# Ejecutar el servidor de Dash
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=3000)
