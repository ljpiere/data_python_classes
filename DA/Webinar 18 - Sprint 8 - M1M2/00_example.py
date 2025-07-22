import requests
import re

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req_text = requests.get(URL).text

print(req_text)