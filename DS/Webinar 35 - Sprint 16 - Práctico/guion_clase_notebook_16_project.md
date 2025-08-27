# Guion de Clase (2h) — Análisis detallado del notebook

*Generado automáticamente el 2025-08-26 23:27*


---

## Agenda sugerida (120 minutos)


- 00:00–10:00 — Contexto y objetivos del proyecto (IMDB reviews, métrica F1 ≥ 0.85).
- 10:00–20:00 — Inicialización y configuración del entorno (imports, `%matplotlib`, `tqdm`).
- 20:00–35:00 — Carga de datos y EDA (distribuciones por año, por película, por puntaje, balance de clases).
- 35:00–45:00 — Procedimiento de evaluación y métricas (F1, matriz de confusión, reporte).
- 45:00–55:00 — Normalización de texto (limpieza HTML, minúsculas, signos, dígitos).
- 55:00–60:00 — División Train/Test.
- 60:00–70:00 — Baseline (DummyClassifier).
- 70:00–90:00 — Modelo 1: TF‑IDF (word 1–2) + Logistic Regression.
- 90:00–100:00 — Modelo 2: TF‑IDF (char 3–5) + Logistic Regression.
- 100:00–110:00 — Modelo 4: TF‑IDF (word 1–2) + Complement Naive Bayes.
- 110:00–120:00 — Modelo 4 (spaCy) + TF‑IDF + LGBMClassifier y conclusiones.


---

## Secciones del notebook

# Descripcipción del proyecto

Film Junky Union, una nueva comunidad vanguardista para los aficionados de las películas clásicas, está desarrollando un sistema para filtrar y categorizar reseñas de películas. Tu objetivo es entrenar un modelo para detectar las críticas negativas de forma automática. Para lograrlo, utilizarás un conjunto de datos de reseñas de películas de IMDB con leyendas de polaridad para construir un modelo para clasificar las reseñas positivas y negativas. Este deberá alcanzar un valor F1 de al menos 0.85.

## Inicialización


**Celda 3 — Código del notebook**


```python
import math

import numpy as np
import pandas as pd

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns

from tqdm.auto import tqdm
```


**Explicación línea por línea:**

- L01: Importa el módulo **math** para usar sus funciones/clases.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Importa el módulo **numpy** y lo renombra como **np** para abreviar su uso.
- L04: Importa el módulo **pandas** y lo renombra como **pd** para abreviar su uso.
- L05: Línea en blanco para separar bloques visualmente.
- L06: Importa el módulo **matplotlib** para usar sus funciones/clases.
- L07: Importa el módulo **matplotlib.pyplot** y lo renombra como **plt** para abreviar su uso.
- L08: Importa el módulo **matplotlib.dates** y lo renombra como **mdates** para abreviar su uso.
- L09: Importa el módulo **seaborn** y lo renombra como **sns** para abreviar su uso.
- L10: Línea en blanco para separar bloques visualmente.
- L11: Desde **tqdm.auto** importa **tqdm** para usarlos directamente.


**Celda 4 — Código del notebook**


```python
%matplotlib inline
%config InlineBackend.figure_format = 'png'
# la siguiente línea proporciona gráficos de mejor calidad en pantallas HiDPI
# %config InlineBackend.figure_format = 'retina'

plt.style.use('seaborn')
```


**Explicación línea por línea:**

- L01: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L02: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L03: Comentario del autor: la siguiente línea proporciona gráficos de mejor calidad en pantallas HiDPI.
- L04: Comentario del autor: %config InlineBackend.figure_format = 'retina'.
- L05: Línea en blanco para separar bloques visualmente.
- L06: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 5 — Código del notebook**


```python
# esto es para usar progress_apply, puedes leer más en https://pypi.org/project/tqdm/#pandas-integration
tqdm.pandas()
```


**Explicación línea por línea:**

- L01: Comentario del autor: esto es para usar progress_apply, puedes leer más en https://pypi.org/project/tqdm/#pandas-integration.
- L02: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.

## Cargar datos


**Celda 7 — Código del notebook**


```python
df_reviews = pd.read_csv('/datasets/imdb_reviews.tsv', sep='\t', dtype={'votes': 'Int64'})
```


**Explicación línea por línea:**

- L01: Carga un archivo CSV/TSV con **pandas.read_csv** especificando separador y/o tipos de datos.

## EDA

Veamos el número de películas y reseñas a lo largo de los años.


**Celda 13 — Código del notebook**


```python
fig, axs = plt.subplots(2, 1, figsize=(16, 8))

ax = axs[0]

dft1 = df_reviews[['tconst', 'start_year']].drop_duplicates() \
    ['start_year'].value_counts().sort_index()
dft1 = dft1.reindex(index=np.arange(dft1.index.min(), max(dft1.index.max(), 2021))).fillna(0)
dft1.plot(kind='bar', ax=ax)
ax.set_title('Número de películas a lo largo de los años')

ax = axs[1]

dft2 = df_reviews.groupby(['start_year', 'pos'])['pos'].count().unstack()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)

dft2.plot(kind='bar', stacked=True, label='#reviews (neg, pos)', ax=ax)

dft2 = df_reviews['start_year'].value_counts().sort_index()
dft2 = dft2.reindex(index=np.arange(dft2.index.min(), max(dft2.index.max(), 2021))).fillna(0)
dft3 = (dft2/dft1).fillna(0)
axt = ax.twinx()
dft3.reset_index(drop=True).rolling(5).mean().plot(color='orange', label='reviews per movie (avg over 5 years)', ax=axt)

lines, labels = axt.get_legend_handles_labels()
ax.legend(lines, labels, loc='upper left')

ax.set_title('Número de reseñas a lo largo de los años')

fig.tight_layout()
```


**Explicación línea por línea:**

- L01: Crea una figura y ejes con **matplotlib.pyplot.subplots**; define tamaño y distribución.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Línea en blanco para separar bloques visualmente.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Cuenta la frecuencia de cada valor en la serie (**value_counts**).
- L07: Reindexa para asegurar un rango continuo y rellena ausentes con 0 (series/años sin datos).
- L08: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L09: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L10: Línea en blanco para separar bloques visualmente.
- L11: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L12: Línea en blanco para separar bloques visualmente.
- L13: Agrupa el DataFrame por una o más columnas y cuenta filas en cada grupo.
- L14: Reindexa para asegurar un rango continuo y rellena ausentes con 0 (series/años sin datos).
- L15: Línea en blanco para separar bloques visualmente.
- L16: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L17: Línea en blanco para separar bloques visualmente.
- L18: Cuenta la frecuencia de cada valor en la serie (**value_counts**).
- L19: Reindexa para asegurar un rango continuo y rellena ausentes con 0 (series/años sin datos).
- L20: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L21: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L22: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L23: Línea en blanco para separar bloques visualmente.
- L24: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L25: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L26: Línea en blanco para separar bloques visualmente.
- L27: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L28: Línea en blanco para separar bloques visualmente.
- L29: Ajusta automáticamente los márgenes para que los elementos no se sobrepongan.

Veamos la distribución del número de reseñas por película con el conteo exacto y KDE (solo para saber cómo puede diferir del conteo exacto)


**Celda 15 — Código del notebook**


```python
fig, axs = plt.subplots(1, 2, figsize=(16, 5))

ax = axs[0]
dft = df_reviews.groupby('tconst')['review'].count() \
    .value_counts() \
    .sort_index()
dft.plot.bar(ax=ax)
ax.set_title('Gráfico de barras de #Reseñas por película')

ax = axs[1]
dft = df_reviews.groupby('tconst')['review'].count()
sns.kdeplot(dft, ax=ax)
ax.set_title('Gráfico KDE de #Reseñas por película')

fig.tight_layout()
```


**Explicación línea por línea:**

- L01: Crea una figura y ejes con **matplotlib.pyplot.subplots**; define tamaño y distribución.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Agrupa el DataFrame por una o más columnas y cuenta filas en cada grupo.
- L05: Cuenta la frecuencia de cada valor en la serie (**value_counts**).
- L06: Ordena por índice (por ejemplo, por año o categoría) para graficar/analizar en orden natural.
- L07: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L08: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L09: Línea en blanco para separar bloques visualmente.
- L10: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L11: Agrupa el DataFrame por una o más columnas y cuenta filas en cada grupo.
- L12: Crea una visualización con **seaborn** (por ejemplo, `kdeplot` para densidad).
- L13: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L14: Línea en blanco para separar bloques visualmente.
- L15: Ajusta automáticamente los márgenes para que los elementos no se sobrepongan.


**Celda 17 — Código del notebook**


```python
df_reviews['pos'].value_counts()
```


**Explicación línea por línea:**

- L01: Cuenta la frecuencia de cada valor en la serie (**value_counts**).


**Celda 18 — Código del notebook**


```python
fig, axs = plt.subplots(1, 2, figsize=(12, 4))

ax = axs[0]
dft = df_reviews.query('ds_part == "train"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de entrenamiento: distribución de puntuaciones')

ax = axs[1]
dft = df_reviews.query('ds_part == "test"')['rating'].value_counts().sort_index()
dft = dft.reindex(index=np.arange(min(dft.index.min(), 1), max(dft.index.max(), 11))).fillna(0)
dft.plot.bar(ax=ax)
ax.set_ylim([0, 5000])
ax.set_title('El conjunto de prueba: distribución de puntuaciones')

fig.tight_layout()
```


**Explicación línea por línea:**

- L01: Crea una figura y ejes con **matplotlib.pyplot.subplots**; define tamaño y distribución.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Cuenta la frecuencia de cada valor en la serie (**value_counts**).
- L05: Reindexa para asegurar un rango continuo y rellena ausentes con 0 (series/años sin datos).
- L06: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L07: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L08: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L09: Línea en blanco para separar bloques visualmente.
- L10: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L11: Cuenta la frecuencia de cada valor en la serie (**value_counts**).
- L12: Reindexa para asegurar un rango continuo y rellena ausentes con 0 (series/años sin datos).
- L13: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L14: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L15: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L16: Línea en blanco para separar bloques visualmente.
- L17: Ajusta automáticamente los márgenes para que los elementos no se sobrepongan.

Distribución de reseñas negativas y positivas a lo largo de los años para dos partes del conjunto de datos


**Celda 21 — Código del notebook**


```python
fig, axs = plt.subplots(2, 2, figsize=(16, 8), gridspec_kw=dict(width_ratios=(2, 1), height_ratios=(1, 1)))

ax = axs[0][0]

dft = df_reviews.query('ds_part == "train"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de entrenamiento: número de reseñas de diferentes polaridades por año')

ax = axs[0][1]

dft = df_reviews.query('ds_part == "train"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de entrenamiento: distribución de diferentes polaridades por película')

ax = axs[1][0]

dft = df_reviews.query('ds_part == "test"').groupby(['start_year', 'pos'])['pos'].count().unstack()
dft.index = dft.index.astype('int')
dft = dft.reindex(index=np.arange(dft.index.min(), max(dft.index.max(), 2020))).fillna(0)
dft.plot(kind='bar', stacked=True, ax=ax)
ax.set_title('El conjunto de prueba: número de reseñas de diferentes polaridades por año')

ax = axs[1][1]

dft = df_reviews.query('ds_part == "test"').groupby(['tconst', 'pos'])['pos'].count().unstack()
sns.kdeplot(dft[0], color='blue', label='negative', kernel='epa', ax=ax)
sns.kdeplot(dft[1], color='green', label='positive', kernel='epa', ax=ax)
ax.legend()
ax.set_title('El conjunto de prueba: distribución de diferentes polaridades por película')

fig.tight_layout()
```


**Explicación línea por línea:**

- L01: Crea una figura y ejes con **matplotlib.pyplot.subplots**; define tamaño y distribución.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Línea en blanco para separar bloques visualmente.
- L05: Agrupa el DataFrame por una o más columnas y cuenta filas en cada grupo.
- L06: Convierte el tipo de datos de una columna/serie con **astype**.
- L07: Reindexa para asegurar un rango continuo y rellena ausentes con 0 (series/años sin datos).
- L08: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L09: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L10: Línea en blanco para separar bloques visualmente.
- L11: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L12: Línea en blanco para separar bloques visualmente.
- L13: Agrupa el DataFrame por una o más columnas y cuenta filas en cada grupo.
- L14: Crea una visualización con **seaborn** (por ejemplo, `kdeplot` para densidad).
- L15: Crea una visualización con **seaborn** (por ejemplo, `kdeplot` para densidad).
- L16: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L17: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L18: Línea en blanco para separar bloques visualmente.
- L19: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L20: Línea en blanco para separar bloques visualmente.
- L21: Agrupa el DataFrame por una o más columnas y cuenta filas en cada grupo.
- L22: Convierte el tipo de datos de una columna/serie con **astype**.
- L23: Reindexa para asegurar un rango continuo y rellena ausentes con 0 (series/años sin datos).
- L24: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L25: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L26: Línea en blanco para separar bloques visualmente.
- L27: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L28: Línea en blanco para separar bloques visualmente.
- L29: Agrupa el DataFrame por una o más columnas y cuenta filas en cada grupo.
- L30: Crea una visualización con **seaborn** (por ejemplo, `kdeplot` para densidad).
- L31: Crea una visualización con **seaborn** (por ejemplo, `kdeplot` para densidad).
- L32: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L33: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L34: Línea en blanco para separar bloques visualmente.
- L35: Ajusta automáticamente los márgenes para que los elementos no se sobrepongan.

## Procedimiento de evaluación

Composición de una rutina de evaluación que se pueda usar para todos los modelos en este proyecto


**Celda 25 — Código del notebook**


```python
import sklearn.metrics as metrics
def evaluate_model(model, train_features, train_target, test_features, test_target):

    eval_stats = {}

    fig, axs = plt.subplots(1, 3, figsize=(20, 6))

    for type, features, target in (('train', train_features, train_target), ('test', test_features, test_target)):

        eval_stats[type] = {}

        pred_target = model.predict(features)
        pred_proba = model.predict_proba(features)[:, 1]

        # F1
        f1_thresholds = np.arange(0, 1.01, 0.05)
        f1_scores = [metrics.f1_score(target, pred_proba>=threshold) for threshold in f1_thresholds]

        # ROC
        fpr, tpr, roc_thresholds = metrics.roc_curve(target, pred_proba)
        roc_auc = metrics.roc_auc_score(target, pred_proba)
        eval_stats[type]['ROC AUC'] = roc_auc

        # PRC
        precision, recall, pr_thresholds = metrics.precision_recall_curve(target, pred_proba)
        aps = metrics.average_precision_score(target, pred_proba)
        eval_stats[type]['APS'] = aps

        if type == 'train':
            color = 'blue'
        else:
            color = 'green'

        # Valor F1
        ax = axs[0]
        max_f1_score_idx = np.argmax(f1_scores)
        ax.plot(f1_thresholds, f1_scores, color=color, label=f'{type}, max={f1_scores[max_f1_score_idx]:.2f} @ {f1_thresholds[max_f1_score_idx]:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(f1_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(f1_thresholds[closest_value_idx], f1_scores[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('threshold')
        ax.set_ylabel('F1')
        ax.legend(loc='lower center')
        ax.set_title(f'Valor F1')

        # ROC
        ax = axs[1]
        ax.plot(fpr, tpr, color=color, label=f'{type}, ROC AUC={roc_auc:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(roc_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(fpr[closest_value_idx], tpr[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.plot([0, 1], [0, 1], color='grey', linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')
        ax.legend(loc='lower center')
        ax.set_title(f'Curva ROC')

        # PRC
        ax = axs[2]
        ax.plot(recall, precision, color=color, label=f'{type}, AP={aps:.2f}')
        # establecer cruces para algunos umbrales
        for threshold in (0.2, 0.4, 0.5, 0.6, 0.8):
            closest_value_idx = np.argmin(np.abs(pr_thresholds-threshold))
            marker_color = 'orange' if threshold != 0.5 else 'red'
            ax.plot(recall[closest_value_idx], precision[closest_value_idx], color=marker_color, marker='X', markersize=7)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel('recall')
        ax.set_ylabel('precision')
        ax.legend(loc='lower center')
        ax.set_title(f'PRC')

        eval_stats[type]['Accuracy'] = metrics.accuracy_score(target, pred_target)
        eval_stats[type]['F1'] = metrics.f1_score(target, pred_target)

    df_eval_stats = pd.DataFrame(eval_stats)
    df_eval_stats = df_eval_stats.round(2)
    df_eval_stats = df_eval_stats.reindex(index=('Accuracy', 'F1', 'APS', 'ROC AUC'))

    print(df_eval_stats)

    return
```


**Explicación línea por línea:**

- L01: Importa el módulo **sklearn.metrics** y lo renombra como **metrics** para abreviar su uso.
- L02: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L03: Línea en blanco para separar bloques visualmente.
- L04: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L05: Línea en blanco para separar bloques visualmente.
- L06: Crea una figura y ejes con **matplotlib.pyplot.subplots**; define tamaño y distribución.
- L07: Línea en blanco para separar bloques visualmente.
- L08: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L09: Línea en blanco para separar bloques visualmente.
- L10: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L11: Línea en blanco para separar bloques visualmente.
- L12: Obtiene predicciones de clase del modelo entrenado.
- L13: Obtiene probabilidades predichas por clase; útil para fijar umbrales.
- L14: Línea en blanco para separar bloques visualmente.
- L15: Comentario del autor: F1.
- L16: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L17: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L18: Línea en blanco para separar bloques visualmente.
- L19: Comentario del autor: ROC.
- L20: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L21: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L22: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L23: Línea en blanco para separar bloques visualmente.
- L24: Comentario del autor: PRC.
- L25: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L26: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L27: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L28: Línea en blanco para separar bloques visualmente.
- L29: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L30: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L31: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L32: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L33: Línea en blanco para separar bloques visualmente.
- L34: Comentario del autor: Valor F1.
- L35: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L36: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L37: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L38: Comentario del autor: establecer cruces para algunos umbrales.
- L39: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L40: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L41: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L42: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L43: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L44: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L45: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L46: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L47: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L48: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L49: Línea en blanco para separar bloques visualmente.
- L50: Comentario del autor: ROC.
- L51: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L52: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L53: Comentario del autor: establecer cruces para algunos umbrales.
- L54: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L55: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L56: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L57: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L58: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L59: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L60: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L61: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L62: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L63: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L64: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L65: Línea en blanco para separar bloques visualmente.
- L66: Comentario del autor: PRC.
- L67: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L68: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L69: Comentario del autor: establecer cruces para algunos umbrales.
- L70: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L71: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L72: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L73: Genera una gráfica con **pandas/matplotlib** a partir del DataFrame/serie actual.
- L74: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L75: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L76: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L77: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L78: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L79: Ajusta una propiedad del eje/gráfico (título, límites, etiquetas, etc.).
- L80: Línea en blanco para separar bloques visualmente.
- L81: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L82: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L83: Línea en blanco para separar bloques visualmente.
- L84: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L85: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L86: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L87: Línea en blanco para separar bloques visualmente.
- L88: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L89: Línea en blanco para separar bloques visualmente.
- L90: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 26 — Código del notebook**


```python
from sklearn.metrics import classification_report, f1_score, confusion_matrix

def evaluate_model(model, X_train, y_train, X_test, y_test, target_names=('neg','pos')):
    """
    Ajusta el modelo, predice y muestra informe de clasificación y F1 macro.
    Devuelve el F1 macro.
    """
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    f1 = f1_score(y_test, y_pred, average='macro')
    print(f"F1 macro: {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=target_names, digits=4))
    print("Matriz de confusión:\n", confusion_matrix(y_test, y_pred, labels=[0,1]))
    return f1
```


**Explicación línea por línea:**

- L01: Desde **sklearn.metrics** importa **classification_report, f1_score, confusion_matrix** para usarlos directamente.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L07: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L08: Entrena/ajusta el modelo/transformador con los datos proporcionados.
- L09: Obtiene predicciones de clase del modelo entrenado.
- L10: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L11: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L12: Imprime métricas por clase (precisión, recall, F1) y macro/micro promedios.
- L13: Muestra la **matriz de confusión** (TP/FP/FN/TN) para analizar errores del modelo.
- L14: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.

## Normalización


**Celda 28 — Código del notebook**


```python
import re, html

TAG_RE = re.compile(r"<[^>]+>")
PUNCT_RE = re.compile(r"[^\w\s]")
DIGIT_RE = re.compile(r"\d+")

def normalize_text(s: str) -> str:
    if not isinstance(s, str):
        s = str(s)
    s = html.unescape(s)
    s = TAG_RE.sub(" ", s)
    s = s.lower()
    s = PUNCT_RE.sub(" ", s)
    s = DIGIT_RE.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

df_reviews['review_norm'] = df_reviews['review'].astype(str).map(normalize_text)
df_reviews.head(3)
```


**Explicación línea por línea:**

- L01: Importa el módulo **re** para usar sus funciones/clases.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Compila una **expresión regular** para reutilizarla eficientemente.
- L04: Compila una **expresión regular** para reutilizarla eficientemente.
- L05: Compila una **expresión regular** para reutilizarla eficientemente.
- L06: Línea en blanco para separar bloques visualmente.
- L07: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L08: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L09: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L10: Convierte entidades HTML (`&amp;`, `&lt;`) a sus caracteres reales.
- L11: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L12: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L13: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L14: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L15: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L16: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L17: Línea en blanco para separar bloques visualmente.
- L18: Convierte el tipo de datos de una columna/serie con **astype**.
- L19: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.

## División entrenamiento / prueba

Por fortuna, todo el conjunto de datos ya está dividido en partes de entrenamiento/prueba; 'ds_part' es el indicador correspondiente.


**Celda 31 — Código del notebook**


```python
df_reviews_train = df_reviews.query('ds_part == "train"').copy()
df_reviews_test = df_reviews.query('ds_part == "test"').copy()

train_target = df_reviews_train['pos']
test_target = df_reviews_test['pos']

print(df_reviews_train.shape)
print(df_reviews_test.shape)
```


**Explicación línea por línea:**

- L01: Filtra filas con **DataFrame.query** usando una expresión estilo SQL.
- L02: Filtra filas con **DataFrame.query** usando una expresión estilo SQL.
- L03: Línea en blanco para separar bloques visualmente.
- L04: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Línea en blanco para separar bloques visualmente.
- L07: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L08: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 32 — Código del notebook**


```python
df_reviews_train.head()
```


**Explicación línea por línea:**

- L01: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.

## Trabajar con modelos

### Modelo 0 - Constante


**Celda 35 — Código del notebook**


```python
# Features dummy (una columna de ceros) para DummyClassifier
X_train_0 = np.zeros((len(df_reviews_train), 1))
X_test_0 = np.zeros((len(df_reviews_test), 1))
```


**Explicación línea por línea:**

- L01: Comentario del autor: Features dummy (una columna de ceros) para DummyClassifier.
- L02: Crea una matriz de ceros; aquí sirve como 'dummy features' para un modelo base.
- L03: Crea una matriz de ceros; aquí sirve como 'dummy features' para un modelo base.


**Celda 36 — Código del notebook**


```python
from sklearn.dummy import DummyClassifier

model_0 = DummyClassifier(strategy='most_frequent', random_state=42)
```


**Explicación línea por línea:**

- L01: Desde **sklearn.dummy** importa **DummyClassifier** para usarlos directamente.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Clasificador de referencia que predice la clase más frecuente (baseline).


**Celda 37 — Código del notebook**


```python
print("=== Modelo 0: Dummy (clase más frecuente) ===")
evaluate_model(model_0, X_train_0, train_target, X_test_0, test_target)
```


**Explicación línea por línea:**

- L01: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L02: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.

### Modelo 1 - NLTK, TF-IDF y LR

TF-IDF


**Celda 41 — Código del notebook**


```python
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorizador TF-IDF (palabras 1-2) sobre review_norm
tfidf_vectorizer_1 = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9,
                                     stop_words='english', strip_accents='unicode')
train_features_1 = tfidf_vectorizer_1.fit_transform(df_reviews_train['review_norm'])
test_features_1  = tfidf_vectorizer_1.transform(df_reviews_test['review_norm'])
```


**Explicación línea por línea:**

- L01: Desde **sklearn.feature_extraction.text** importa **TfidfVectorizer** para usarlos directamente.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Comentario del autor: Vectorizador TF-IDF (palabras 1-2) sobre review_norm.
- L04: Configura un **TfidfVectorizer** (rango de n-gramas, stopwords, etc.) para convertir texto a vectores.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Ajusta el transformador al conjunto de entrenamiento y transforma esos datos a la representación numérica.
- L07: Aplica el transformador previamente ajustado a nuevos datos (por ejemplo, el conjunto de prueba).


**Celda 42 — Código del notebook**


```python
from sklearn.linear_model import LogisticRegression

# Regresión Logística con balance de clases
model_1 = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced', random_state=42)

print("=== Modelo 1: TF-IDF (word 1-2) + LogisticRegression ===")
```


**Explicación línea por línea:**

- L01: Desde **sklearn.linear_model** importa **LogisticRegression** para usarlos directamente.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Comentario del autor: Regresión Logística con balance de clases.
- L04: Crea un modelo de **Regresión Logística**; parámetros como `class_weight='balanced'` manejan desbalance.
- L05: Línea en blanco para separar bloques visualmente.
- L06: Crea un modelo de **Regresión Logística**; parámetros como `class_weight='balanced'` manejan desbalance.

### Modelo 2 - TF-IDF (char 3-5) y Regresión Logística


**Celda 45 — Código del notebook**


```python
# Vectorizador 2: n-gramas de caracteres (3-5)
tfidf_vectorizer_2 = TfidfVectorizer(analyzer='char', ngram_range=(3,5), min_df=2)
train_features_2 = tfidf_vectorizer_2.fit_transform(df_reviews_train['review_norm'])
test_features_2  = tfidf_vectorizer_2.transform(df_reviews_test['review_norm'])

# Modelo: LR (usa predict_proba para compatibilidad con la sección "Mis reseñas")
model_2 = LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42)

print("=== Modelo 2: TF-IDF (char 3-5) + LogisticRegression ===")
evaluate_model(model_2, train_features_2, train_target, test_features_2, test_target)
```


**Explicación línea por línea:**

- L01: Línea en blanco para separar bloques visualmente.
- L02: Comentario del autor: Vectorizador 2: n-gramas de caracteres (3-5).
- L03: Configura un **TfidfVectorizer** (rango de n-gramas, stopwords, etc.) para convertir texto a vectores.
- L04: Ajusta el transformador al conjunto de entrenamiento y transforma esos datos a la representación numérica.
- L05: Aplica el transformador previamente ajustado a nuevos datos (por ejemplo, el conjunto de prueba).
- L06: Línea en blanco para separar bloques visualmente.
- L07: Comentario del autor: Modelo: LR (usa predict_proba para compatibilidad con la sección "Mis reseñas").
- L08: Crea un modelo de **Regresión Logística**; parámetros como `class_weight='balanced'` manejan desbalance.
- L09: Línea en blanco para separar bloques visualmente.
- L10: Crea un modelo de **Regresión Logística**; parámetros como `class_weight='balanced'` manejan desbalance.
- L11: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 46 — Código del notebook**


```python
evaluate_model(model_1, train_features_1, train_target, test_features_1, test_target)
```


**Explicación línea por línea:**

- L01: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 47 — Código del notebook**


```python
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    except Exception as e:
        print("No se pudo cargar 'en_core_web_sm':", e)
        nlp = None
except Exception as e:
    print("spaCy no disponible:", e)
    nlp = None
```


**Explicación línea por línea:**

- L01: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L02: Importa el módulo **spacy** para usar sus funciones/clases.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Carga un **pipeline de spaCy** (o crea uno vacío como fallback) para tokenizar/lematizar.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L07: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L08: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L09: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L10: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 48 — Código del notebook**


```python
def text_preprocessing_3(text):
    # Si spaCy no está disponible, devolvemos el texto normalizado tal cual
    t = text if isinstance(text, str) else str(text)
    if nlp is None:
        return t
    doc = nlp(t)
    tokens = [token.lemma_ for token in doc]
    return " ".join(tokens)
```


**Explicación línea por línea:**

- L01: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L02: Comentario del autor: Si spaCy no está disponible, devolvemos el texto normalizado tal cual.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L07: Extrae la **lema** (forma básica) de cada token con spaCy.
- L08: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 49 — Código del notebook**


```python
# Vectorizador 3: TF-IDF palabras 1-2 sobre texto lematizado (si está spaCy)
tfidf_vectorizer_3 = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9,
                                     stop_words='english', strip_accents='unicode')
train_features_3 = tfidf_vectorizer_3.fit_transform(df_reviews_train['review_norm'].apply(text_preprocessing_3))
test_features_3  = tfidf_vectorizer_3.transform(df_reviews_test['review_norm'].apply(text_preprocessing_3))
```


**Explicación línea por línea:**

- L01: Comentario del autor: Vectorizador 3: TF-IDF palabras 1-2 sobre texto lematizado (si está spaCy).
- L02: Configura un **TfidfVectorizer** (rango de n-gramas, stopwords, etc.) para convertir texto a vectores.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Ajusta el transformador al conjunto de entrenamiento y transforma esos datos a la representación numérica.
- L05: Aplica el transformador previamente ajustado a nuevos datos (por ejemplo, el conjunto de prueba).

### Modelo 4 - TF-IDF (word 1-2) y Complement Naive Bayes


**Celda 51 — Código del notebook**


```python
from sklearn.naive_bayes import ComplementNB

# Reutilizamos un vectorizador de palabras (1-2)
tfidf_vectorizer_4 = TfidfVectorizer(ngram_range=(1,2), min_df=2, max_df=0.9,
                                     stop_words='english', strip_accents='unicode')
train_features_4 = tfidf_vectorizer_4.fit_transform(df_reviews_train['review_norm'])
test_features_4  = tfidf_vectorizer_4.transform(df_reviews_test['review_norm'])

model_4 = ComplementNB()

print("=== Modelo 4: TF-IDF (word 1-2) + ComplementNB ===")
evaluate_model(model_4, train_features_4, train_target, test_features_4, test_target)
```


**Explicación línea por línea:**

- L01: Desde **sklearn.naive_bayes** importa **ComplementNB** para usarlos directamente.
- L02: Línea en blanco para separar bloques visualmente.
- L03: Comentario del autor: Reutilizamos un vectorizador de palabras (1-2).
- L04: Configura un **TfidfVectorizer** (rango de n-gramas, stopwords, etc.) para convertir texto a vectores.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Ajusta el transformador al conjunto de entrenamiento y transforma esos datos a la representación numérica.
- L07: Aplica el transformador previamente ajustado a nuevos datos (por ejemplo, el conjunto de prueba).
- L08: Línea en blanco para separar bloques visualmente.
- L09: Crea un clasificador **Complement Naive Bayes**, robusto en datos de texto y clases desbalanceadas.
- L10: Línea en blanco para separar bloques visualmente.
- L11: Crea un clasificador **Complement Naive Bayes**, robusto en datos de texto y clases desbalanceadas.
- L12: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.


**Celda 52 — Código del notebook**


```python
model_3 = LogisticRegression(max_iter=2000, solver='liblinear', class_weight='balanced', random_state=42)
print("=== Modelo 3: spaCy (lemmas*) + TF-IDF (word 1-2) + LogisticRegression ===")
evaluate_model(model_3, train_features_3, train_target, test_features_3, test_target)
```


**Explicación línea por línea:**

- L01: Crea un modelo de **Regresión Logística**; parámetros como `class_weight='balanced'` manejan desbalance.
- L02: Crea un modelo de **Regresión Logística**; parámetros como `class_weight='balanced'` manejan desbalance.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.

### Modelo 4 - spaCy, TF-IDF y LGBMClassifier


**Celda 56 — Código del notebook**


```python
from lightgbm import LGBMClassifier
```


**Explicación línea por línea:**

- L01: Desde **lightgbm** importa **LGBMClassifier** para usarlos directamente.


**Celda 57 — Código del notebook**


```python
# --- Imports
import re, spacy
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix
from lightgbm import LGBMClassifier

# --- Config
TEXT_COL = "review"          # <-- cambia al nombre real de tu columna de texto
SPACY_MODEL = "en_core_web_sm"   # o "es_core_news_sm" si es español

# --- Tokenizador/Lematizador con spaCy
try:
    nlp = spacy.load(SPACY_MODEL, disable=["parser","ner","textcat"])
except OSError:
    nlp = spacy.blank("en")  # fallback si no está instalado

def spacy_lemmas(text):
    text = re.sub(r"\s+", " ", str(text)).strip().lower()
    doc = nlp(text)
    toks = []
    for t in doc:
        if t.is_space or t.is_punct or t.like_num or t.is_quote or t.is_stop:
            continue
        toks.append(t.lemma_ if t.lemma_ not in ("", "-PRON-") else t.text)
    return toks

# --- 1) TF-IDF independiente
tfidf = TfidfVectorizer(
    tokenizer=spacy_lemmas,
    preprocessor=None,
    lowercase=False,
    ngram_range=(1,2),
    min_df=3,
    max_df=0.9,
    sublinear_tf=True,
    max_features=20000
)

X_train = tfidf.fit_transform(df_reviews_train[TEXT_COL])
X_test  = tfidf.transform(df_reviews_test[TEXT_COL])
```


**Explicación línea por línea:**

- L01: Comentario del autor: --- Imports.
- L02: Importa el módulo **re** para usar sus funciones/clases.
- L03: Importa el módulo **numpy** y lo renombra como **np** para abreviar su uso.
- L04: Desde **sklearn.feature_extraction.text** importa **TfidfVectorizer** para usarlos directamente.
- L05: Desde **sklearn.metrics** importa **f1_score, accuracy_score, classification_report, confusion_matrix** para usarlos directamente.
- L06: Desde **lightgbm** importa **LGBMClassifier** para usarlos directamente.
- L07: Línea en blanco para separar bloques visualmente.
- L08: Comentario del autor: --- Config.
- L09: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L10: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L11: Línea en blanco para separar bloques visualmente.
- L12: Comentario del autor: --- Tokenizador/Lematizador con spaCy.
- L13: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L14: Carga un **pipeline de spaCy** (o crea uno vacío como fallback) para tokenizar/lematizar.
- L15: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L16: Carga un **pipeline de spaCy** (o crea uno vacío como fallback) para tokenizar/lematizar.
- L17: Línea en blanco para separar bloques visualmente.
- L18: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L19: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L20: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L21: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L22: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L23: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L24: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L25: Extrae la **lema** (forma básica) de cada token con spaCy.
- L26: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L27: Línea en blanco para separar bloques visualmente.
- L28: Comentario del autor: --- 1) TF-IDF independiente.
- L29: Configura un **TfidfVectorizer** (rango de n-gramas, stopwords, etc.) para convertir texto a vectores.
- L30: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L31: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L32: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L33: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L34: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L35: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L36: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L37: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L38: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L39: Línea en blanco para separar bloques visualmente.
- L40: Ajusta el transformador al conjunto de entrenamiento y transforma esos datos a la representación numérica.
- L41: Aplica el transformador previamente ajustado a nuevos datos (por ejemplo, el conjunto de prueba).


**Celda 58 — Código del notebook**


```python
# --- 2) LGBMClassifier
clf = LGBMClassifier(
    objective="binary",
    class_weight="balanced",
    random_state=42,
    n_estimators=600,
    learning_rate=0.05,
    num_leaves=63,
    min_child_samples=20,
    subsample=0.9,
    colsample_bytree=0.9,
    n_jobs=-1
)
clf.fit(X_train, train_target.astype(int))

# --- 3) Evaluación
y_proba = clf.predict_proba(X_test)[:, 1]
y_pred  = (y_proba >= 0.50).astype(int)

print(f"F1: {f1_score(test_target, y_pred):.4f} | Acc: {accuracy_score(test_target, y_pred):.4f}")
print("\nReporte:\n", classification_report(test_target, y_pred, digits=4))
print("Matriz de confusión:\n", confusion_matrix(test_target, y_pred))
```


**Explicación línea por línea:**

- L01: Comentario del autor: --- 2) LGBMClassifier.
- L02: Crea un **LightGBM** (árboles potenciados por gradiente) para clasificación binaria.
- L03: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L04: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L05: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L06: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L07: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L08: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L09: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L10: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L11: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L12: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L13: Ejecuta una operación acorde al contexto; se explicará durante la clase si genera dudas.
- L14: Convierte el tipo de datos de una columna/serie con **astype**.
- L15: Línea en blanco para separar bloques visualmente.
- L16: Comentario del autor: --- 3) Evaluación.
- L17: Obtiene probabilidades predichas por clase; útil para fijar umbrales.
- L18: Convierte el tipo de datos de una columna/serie con **astype**.
- L19: Línea en blanco para separar bloques visualmente.
- L20: Calcula y muestra la métrica **F1** (harmónica de precisión y exhaustividad).
- L21: Imprime métricas por clase (precisión, recall, F1) y macro/micro promedios.
- L22: Muestra la **matriz de confusión** (TP/FP/FN/TN) para analizar errores del modelo.

## Conclusiones

**Resumen de hallazgos**  
- Los modelos lineales con TF–IDF alcanzan típicamente F1 ≥ 0.85 en IMDB.  
- En particular, TF–IDF de **caracteres (3–5) + LR** y TF–IDF de **palabras (1–2) + LR** suelen rendir muy bien.  
- **ComplementNB** es veloz y competitivo; a veces queda justo por debajo de LR, pero ofrece buena calibración de probabilidad.
- Si el conjunto presenta ligero desequilibrio, usar `class_weight='balanced'` ayuda a subir *recall* de la clase minoritaria.

**Recomendación**  
- Seleccionar el modelo con mejor F1 en *test* (esperado ≥ 0.85).  
- Guardar el *pipeline* (vectorizador + clasificador) con `joblib` para despliegue.  
- Añadir una validación adicional con *cross-validation* para robustez y *threshold tuning* si se optimiza F1 de la clase negativa.


---

## Notas didácticas para el/la docente


- **Datos**: la columna `review` contiene el texto; `pos` es la etiqueta binaria (0=negativa, 1=positiva); `ds_part` separa train/test.
- **Métricas**: usar **F1** como métrica objetivo (evita optimizar solo accuracy en caso de desbalance).
- **Vectorización**: comparar TF‑IDF por **palabras** vs por **caracteres**; discutir cuándo es útil cada enfoque.
- **Modelos**: partir de **DummyClassifier** (baseline), luego **LogisticRegression** y **ComplementNB**; cerrar con **LightGBM** sobre TF‑IDF.
- **Preprocesamiento**: demostrar impacto de normalizar el texto (`normalize_text`) y, si está disponible, **lematización** con `spaCy`.
- **Umbral**: comentar el efecto de variar umbral de clasificación cuando se usa `predict_proba` (p.ej. 0.5 vs 0.6) y su impacto en F1/Recall/Precision.
- **Reproducibilidad**: fijar `random_state`; documentar versiones de librerías si es posible.
