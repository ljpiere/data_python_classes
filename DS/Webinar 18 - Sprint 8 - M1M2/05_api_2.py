import requests

city = 'Cancún'

BASE_URL = f'https://wttr.in/{city}'
# URL para el método get()

params = {"m":"", "format":3}


response = requests.get(BASE_URL, params=params)
print(response.text)