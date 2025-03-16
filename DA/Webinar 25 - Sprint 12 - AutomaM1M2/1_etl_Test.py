#!/usr/bin/python

# Importar librerías
import pandas as pd
from sqlalchemy import create_engine

# Definir los parámetros para conectarse a la base de datos
# los puedes solicitar al administrador de la base de datos.
db_config = {'user': 'arq',         # nombre de usuario
             'pwd': 'password', # contraseña
             'host': 'localhost',       # dirección del servidor
             'port': 5432,              # puerto de conexión
             'db': 'bd'}             # nombre de la base de datos

# Crear string de conexión de la base de datos. 
connection_string = 'postgresql://{}:{}@{}:{}/{}'.format(db_config['user'],
                                                                     db_config['pwd'],
                                                                       db_config['host'],
                                                                       db_config['port'],
                                                                       db_config['db'])
# Conectarse a la base de datos.
engine = create_engine(connection_string)

query = ''' select * 
            from public."event"
        '''

# Ejecutar la consulta y almacenar el resultado
# en el DataFrame.
# SQLAlchemy automáticamente dará a las columnas
# los mismos nombres que tienen en la tabla de la base de datos. Solo tendremos que
# especificar la columna de índice con index_col.
# Ej: , index_col = 'game_id'
data_raw = pd.io.sql.read_sql(query, con = engine)

print(data_raw.head(5))