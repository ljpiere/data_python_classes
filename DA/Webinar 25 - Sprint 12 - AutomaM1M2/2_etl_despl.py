import sys
import getopt
import pandas as pd
import re
from sqlalchemy import create_engine, text
from sqlalchemy.exc import OperationalError

def parse_arguments():
    """
    Analiza los argumentos de la línea de comandos para la ruta del archivo.
    
    Devuelve:
        file_path (str): La ruta del archivo proporcionada por el usuario.
    """
    unixOptions = "f:"
    gnuOptions = ["file="]

    fullCmdArguments = sys.argv
    argumentList = fullCmdArguments[1:]  # No incluir el nombre del script

    try:
        arguments, values = getopt.getopt(argumentList, unixOptions, gnuOptions)
    except getopt.error as err:
        print(str(err))
        sys.exit(2)

    file_path = ''

    for currentArgument, currentValue in arguments:
        if currentArgument in ("-f", "--file"):
            file_path = currentValue
    
    return file_path

def extract_year_from_path(file_path):
    # Divide la ruta del archivo en partes
    year = file_path.split('/')[-1].split('.')[0][-4:]
    
    return year

def data_already_exists(engine, table_name, year):
    try:
        with engine.connect() as connection:
            # Consulta la tabla nueva
            query = text(f'SELECT * FROM {table_name} WHERE ANO_EGRESO={year}')
            result = connection.execute(query)
            exists = result.fetchone() is not None
    except OperationalError as e:
            exists = False
    return exists

def load_data(file_path):
    df = pd.read_csv(file_path, encoding='latin1', delimiter=';')
    return df

def preprocess_data(df, threshold=0.5):
    """
    Preprocesa el DataFrame eliminando las filas en las que la mayoría de las columnas contienen el caracter '*'.
    
    Args:
        df (pd.DataFrame): El DataFrame de entrada.
        threshold (float): La proporción de columnas que pueden contener '*' antes de que la fila se elimine.
    
    Devuelve:
        pd.DataFrame: El DataFrame limpio.
    """

    ## Elimina filas ausentes
        ## Calcula el número de columnas
    num_columns = df.shape[1]

    # Determina el número de '*' permitidos con base en el umbral especificado (threshold).
    allowed_stars = int(num_columns * threshold)

    # Filtra las filas en las que el número de '*' sobrepasa el umbral especificado permitido.
    cleaned_df = df[df.apply(lambda x: (x == '*').sum() <= allowed_stars, axis=1)]
    
    # Formato de los datos
    cleaned_df.loc[:,'COMUNA_RESIDENCIA'] = cleaned_df['COMUNA_RESIDENCIA'].astype(int)
    cleaned_df.loc[:,'REGION_RESIDENCIA'] = cleaned_df['REGION_RESIDENCIA'].astype(int)
    cleaned_df.loc[:,'ANO_EGRESO'] = cleaned_df['ANO_EGRESO'].astype(int)
    
    # renombre las columnas
    new_column_names = ['PERTENENCIA_ESTABLECIMIENTO_SALUD', 'SEXO', 'GRUPO_EDAD', 'ETNIA',
       'GLOSA_PAIS_ORIGEN', 'COMUNA_RESIDENCIA', 'GLOSA_COMUNA_RESIDENCIA',
       'REGION_RESIDENCIA', 'GLOSA_REGION_RESIDENCIA', 'PREVISION',
       'GLOSA_PREVISION', 'ANO_EGRESO', 'DIAG1', 'DIAG2', 'DIAS_ESTADA',
       'CONDICION_EGRESO', 'INTERV_Q', 'PROCED']
    old_column_names = cleaned_df.columns
    
    column_mapping = dict(zip(old_column_names, new_column_names))
    cleaned_df.rename(columns=column_mapping, inplace=True)

    return cleaned_df

def create_db_engine(db_name):
    #connection_string = f'sqlite:///{db_name}'
    #engine = create_engine(connection_string)
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
    print(f'[INFO]: Connection Checked: {connection_string}')
    return engine

def save_to_database(df, engine, table_name):
    df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
    
def validate_data(engine, table_name):  
    with engine.connect() as connection:
        # Consulta la tabla nueva
        query = text(f'SELECT ANO_EGRESO, count(*) FROM {table_name} GROUP BY ANO_EGRESO')
        result = connection.execute(query)

        rows = result.fetchall()
        for row in rows[:100]:
            print(row)                    
      
if __name__ == "__main__":
    # Analizar la ruta del archivo a partir de los argumentos de la línea de comandos
    file_path = parse_arguments()
    table_name = 'egresos_pacientes'
    
    # Carga la base de datos en tu sistema
    engine = create_db_engine('database/ministerio_de_salud_chile.db')   
    print('[INFO]: Database connection')
    
    if file_path:    
        print(f"File Path: {file_path}")
        year = extract_year_from_path(file_path)
         
        # Carga tu archivo csv en un DataFrame de Pandas     
        raw_data = load_data(file_path)
        print('[INFO]: Load data as a pandas dataframe')
        
        # Comprueba si los datos ya están en la base de datos
        if data_already_exists(engine, table_name, year):
            print(f"The data already exists in the database. No se realizó ninguna acción.")        
        else:
            # Preprocesa los datos:
            raw_data = preprocess_data(raw_data)
            print('[INFO]: Preprocess data')
            
            # Guarda los datos en una tabla nueva dentro de la base de datos que ya existe   
            save_to_database(raw_data, engine, table_name)
            print('[INFO]: Loads data into DB')
    else:
        print('No path was provided')
    
    validate_data(engine, table_name)