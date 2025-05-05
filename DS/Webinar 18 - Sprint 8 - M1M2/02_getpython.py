import requests
import re

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req_text = requests.get(URL).text
found_products = re.findall('Horizon[ \w\-%]+', req_text)
print(len(found_products))
print(found_products)