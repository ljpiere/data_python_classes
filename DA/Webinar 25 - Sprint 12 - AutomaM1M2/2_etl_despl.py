#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import re
import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError

# ---------------------------
# 1) Parámetros / conexión
# ---------------------------
def parse_args():
    p = argparse.ArgumentParser(description="ETL minimal CSV → PostgreSQL")
    p.add_argument("-f", "--file", required=True, help="Ruta al archivo CSV")
    p.add_argument("--sep", default=";", help="Separador del CSV (default: ;) ")
    p.add_argument("--enc", default="latin1", help="Encoding del CSV (default: latin1)")
    p.add_argument("--table", default="egresos_pacientes", help="Tabla destino")
    return p.parse_args()

def pg_engine():
    # Usa env vars si existen; si no, defaults a airflow/airflow/airflow@localhost:5432
    user = os.getenv("PGUSER", "airflow")
    pwd  = os.getenv("PGPASSWORD", "airflow")
    host = os.getenv("PGHOST", "localhost")
    port = os.getenv("PGPORT", "5432")
    db   = os.getenv("PGDATABASE", "airflow")
    url  = f"postgresql://{user}:{pwd}@{host}:{port}/{db}"
    print(f"[INFO] Conectando a {url}")
    return create_engine(url)

# ---------------------------
# 2) Extract
# ---------------------------
def extract(csv_path: str, sep: str, enc: str) -> pd.DataFrame:
    print(f"[INFO] Leyendo CSV: {csv_path}")
    return pd.read_csv(csv_path, sep=sep, encoding=enc)

def year_from_filename(path: str) -> int | None:
    # Busca un año de 4 dígitos en el nombre del archivo
    m = re.search(r"(\d{4})", os.path.basename(path))
    return int(m.group(1)) if m else None

# ---------------------------
# 3) Transform (ligero)
# ---------------------------
def transform(df: pd.DataFrame, fallback_year: int | None) -> pd.DataFrame:
    print("[INFO] Transform: limpieza básica")
    # Normaliza nombres de columnas: minúsculas y sin espacios
    df.columns = (
        df.columns.str.strip()
                  .str.lower()
                  .str.replace(" ", "_")
                  .str.replace(r"[^a-z0-9_]", "", regex=True)
    )

    # Reemplaza '*' por NaN y elimina filas con demasiados NaN
    df = df.replace("*", pd.NA)
    df = df.dropna(axis=0, thresh=max(1, int(df.shape[1] * 0.5)))  # al menos 50% de no-nulos

    # Asegura columna de año
    if "ano_egreso" not in df.columns:
        if fallback_year is None:
            print("[WARN] No se detectó año en el nombre del archivo y falta 'ano_egreso' en el CSV.")
            df["ano_egreso"] = pd.NA
        else:
            df["ano_egreso"] = fallback_year

    # Intenta castear algunas columnas típicas si existen
    INT_COLS = ["comuna_residencia", "region_residencia", "ano_egreso", "dias_estada"]
    for c in INT_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Renombres opcionales (ajusta si tu CSV trae otros nombres)
    RENAME_MAP = {
        "glosa_comuna_residencia": "comuna_nombre",
        "glosa_region_residencia": "region_nombre",
    }
    df = df.rename(columns={k: v for k, v in RENAME_MAP.items() if k in df.columns})

    # Ejemplo de recorte de columnas para que quede liviano (mantén solo algunas si existen)
    KEEP = [
        "sexo", "grupo_edad", "etnia", "comuna_residencia", "comuna_nombre",
        "region_residencia", "region_nombre", "prevision", "ano_egreso",
        "diag1", "dias_estada",
    ]
    df = df[[c for c in KEEP if c in df.columns]].copy()

    print(f"[INFO] Filas tras limpieza: {len(df)}  |  Columnas: {len(df.columns)}")
    return df

# ---------------------------
# 4) Load
# ---------------------------
def load(df: pd.DataFrame, engine, table: str):
    print(f"[INFO] Cargando en tabla: {table}")
    df.to_sql(table, engine, if_exists="append", index=False)
    print("[INFO] Carga completada")

# ---------------------------
# 5) Validate
# ---------------------------
def validate(engine, table: str):
    print("[INFO] Validación: conteo por ano_egreso")
    try:
        with engine.connect() as con:
            res = con.execute(text(f"""
                SELECT ano_egreso, COUNT(*) AS filas
                FROM {table}
                GROUP BY ano_egreso
                ORDER BY ano_egreso
            """))
            rows = res.fetchall()
            if not rows:
                print("  (sin filas todavía)")
            for r in rows:
                print(f"  año={r[0]}  filas={r[1]}")
    except SQLAlchemyError as e:
        print(f"[ERROR] Validación falló: {e}")

# ---------------------------
# Main
# ---------------------------
def main():
    args = parse_args()
    engine = pg_engine()

    df_raw = extract(args.file, args.sep, args.enc)
    yr = year_from_filename(args.file)
    df_t = transform(df_raw, yr)
    load(df_t, engine, args.table)
    validate(engine, args.table)

if __name__ == "__main__":
    main()
