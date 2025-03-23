import requests

city = 'Rome'

BASE_URL = f'https://wttr.in/{city}'
# URL para el método get()

params = {"m":""}


response = requests.get(BASE_URL, params=params)
print(response.text)