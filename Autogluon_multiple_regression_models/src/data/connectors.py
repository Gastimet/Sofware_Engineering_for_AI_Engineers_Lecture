# /workspace/src/data/connectors.py
from __future__ import annotations
import polars as pl
import pandas as pd
import requests
from typing import Optional, Dict, Any

# Basit ve hızlı connector’lar. Hepsi Polars DataFrame döndürür.

class CSVConnector:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pl.DataFrame:
        return pl.read_csv(self.path)

class PostgresConnector:
    def __init__(self, conn_str: str, sql: str):
        self.conn_str = conn_str
        self.sql = sql

    def load(self) -> pl.DataFrame:
        import sqlalchemy as sa
        engine = sa.create_engine(self.conn_str)
        with engine.connect() as con:
            df = pd.read_sql(self.sql, con)
        return pl.from_pandas(df)

class RESTConnector:
    def __init__(
        self,
        url: str,
        headers: Optional[Dict[str, str]] = None,
        params: Optional[Dict[str, Any]] = None,
    ):
        self.url = url
        self.headers = headers or {}
        self.params = params or {}

    def load(self) -> pl.DataFrame:
        r = requests.get(self.url, headers=self.headers, params=self.params, timeout=30)
        r.raise_for_status()
        data = r.json()
        # Beklenen JSON -> kayıt listesi
        if isinstance(data, dict):
            # olası tek kök anahtarı al
            if len(data) == 1 and isinstance(list(data.values())[0], list):
                data = list(data.values())[0]
            else:
                data = [data]
        return pl.from_dicts(data)
