import pandas as pd
import requests  # Importa la librería para enviar solicitudes al servidor
from bs4 import BeautifulSoup  # Importa la librería para analizar la página web

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req = requests.get(URL)  # solicitud GET
soup = BeautifulSoup(req.text, 'lxml')

name_products = []  # Lista donde se almacenan los nombres de los productos
for row in soup.find_all(
    'p', attrs={'class': 't754__title t-name t-name_md js-product-name'}
):
    name_products.append(row.text)
price = []  # Lista donde se almacenan los precios de los productos
for row in soup.find_all(
    'p', attrs={'class': 't754__price-value js-product-price'}
):
    price.append(row.text)
products_data = (
    pd.DataFrame()
)  # DataFrame con los datos de nombres de productos y precios
products_data['name'] = name_products
products_data['price'] = price
print(products_data.head(5))