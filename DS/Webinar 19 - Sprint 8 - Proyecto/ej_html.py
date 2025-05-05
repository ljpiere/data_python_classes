import pandas as pd

# URL del sitio web con los datos del clima en Chicago en noviembre de 2017
url = "https://practicum-content.s3.us-west-1.amazonaws.com/data-analyst-eng/moved_chicago_weather_2017.html"

# Extrae la tabla que tiene el atributo id="weather_records"
weather_records = pd.read_html(url, attrs={"id": "weather_records"})[0]

# Imprime el DataFrame completo
print(weather_records)
