conteo = 0
lista_numeros = [2,6,9,0]
for i in lista_numeros:
    conteo -= i

print(conteo)

#---------------------------------------------------------------------
# Suponiendo que el dataframe que queremos
# revisar se llama ords

for i in order_products.columns:
    calculo = order_products[i].isna().sum()
    print("Nombre columna: ", i, ". Cantidad de vacíos:", calculo)
