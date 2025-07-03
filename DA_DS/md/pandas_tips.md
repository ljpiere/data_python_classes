### Uso de `.loc[]` en Pandas

El método `.loc[]` permite **seleccionar y asignar** datos en un DataFrame utilizando **etiquetas** (índices y nombres de columna), en lugar de posiciones numéricas.

#### Sintaxis general
```python
df.loc[filtro_de_filas, selección_de_columnas]
```

#### Parámetros

- **`filtro_de_filas`**  
  - **Etiqueta única**: una sola etiqueta de índice (`df.loc[5]`).  
  - **Lista de etiquetas**: varias etiquetas (`df.loc[[1, 3, 5]]`).  
  - **Condición booleana**: mascarilla que devuelve `True/False` por fila (`df.loc[df["edad"] > 30]`).

- **`selección_de_columnas`** *(opcional)*  
  - **Etiqueta única**: un nombre de columna (`df.loc[:, "nombre"]`).  
  - **Lista de etiquetas**: varias columnas (`df.loc[:, ["nombre", "edad"]]`).  
  - **Rango de etiquetas**: mediante slicing (`df.loc[:, "col1":"col4"]`).  
  - Si se omite, se seleccionan todas las columnas (`df.loc[filtro_de_filas]`).

#### Ejemplos

```python
# Filtrar por condición y ver todas las columnas:
df.loc[df["ventas"] > 1000]

# Filtrar filas y ver solo ciertas columnas:
df.loc[df["categoria"] == "Electrónica", ["producto", "precio"]]

# Selección por etiquetas de fila y columna:
df.loc["fila_10", "columna_A"]

# Asignación de valores a un subconjunto:
df.loc[df["puntuacion"] < 50, "estado"] = "revisar"
```

---

### Slicing por posición en Pandas

```python
df[9:20]
```

- **Qué hace**: Selecciona las filas cuya posición (integer-location) va de la 9 (incluida) hasta la 20 (excluida).  
- **Equivalente a**:  
  ```python
  df.iloc[9:20]
  ```  
- **Uso típico**: Extraer un bloque continuo de filas para inspección o procesamiento.

---

### Uso de `.iloc[]` en Pandas

El método `.iloc[]` permite **seleccionar** datos por **posiciones enteras** (integer-location), ignorando las etiquetas del índice.

#### Sintaxis general
```python
df.iloc[fila_inicio:fila_fin, columna_inicio:columna_fin]
```

#### Ejemplos

```python
# Quinta fila (0-based) en todas las columnas:
df.iloc[5]

# Filas de la 3 a la 8 (excluye la 8):
df.iloc[2:8]

# Segunda columna en todas las filas:
df.iloc[:, 1]

# Filas 4-6 y columnas 1-2:
df.iloc[3:6, 0:2]
```

---

### Operaciones de filtrado en Pandas

```python
df[df['Album'] == '1989']
```
- Selecciona todas las filas donde la columna **Album** es igual a `'1989'`.

```python
df[(df['Spotify Streams (millions)'] >= 900) & (df['Spotify Streams (millions)'] <= 1000)]
```
- Selecciona filas donde **Spotify Streams (millions)** está entre 900 y 1000 (inclusive).  
- Utiliza el operador **&** para combinar condiciones booleanas y paréntesis para agrupar cada condición.

> Estas técnicas se conocen como **indexación booleana** o **filtrado condicional**, y devuelven un DataFrame con solo las filas que cumplen los criterios especificados.

---

### Operaciones de agrupamiento con `groupby` en Pandas

- **Qué hace**: Agrupa filas del DataFrame según valores de una o varias columnas, devolviendo un objeto `GroupBy` sobre el que se pueden aplicar funciones de agregación, transformación o filtrado.

- **Sintaxis básica**:
```python
df.groupby('columna')
df.groupby(['col1', 'col2'])
```

- **Métodos comunes**:
  - `agg()` o `aggregate()`: aplica funciones agregadas.
  - `mean()`, `sum()`, `count()`, `min()`, `max()`.
  - `size()`: devuelve el tamaño de cada grupo.
  - `get_group(nombre_grupo)`: obtiene el subconjunto correspondiente a un grupo específico.

- **Ejemplos**:
```python
# 1. Suma de ventas por categoría
df.groupby('categoria')['ventas'].sum()

# 2. Estadísticas descriptivas por región
df.groupby('region').describe()

# 3. Número de registros por usuario
df.groupby('usuario').size()

# 4. Aplicar múltiples agregaciones a ventas
df.groupby('categoria')['ventas'].agg(['sum', 'mean', 'max'])
```
