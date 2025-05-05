import requests
import json

city = 'Paris'

BASE_URL = f'https://wttr.in/{city}'
# URL para el método get() 

PARAM={"format":"j1"}


response = requests.get(BASE_URL, params = PARAM)

response_parsed = json.loads(response.text)
fact_weather = response_parsed['current_condition']
print(json.dumps(fact_weather))