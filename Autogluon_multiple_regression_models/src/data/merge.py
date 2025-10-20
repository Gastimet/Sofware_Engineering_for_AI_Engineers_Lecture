# /workspace/src/data/merge.py
import polars as pl
from typing import List

def join_many(dfs: List[pl.DataFrame], on: List[str], how: str = "left", coalesce: bool = True) -> pl.DataFrame:
    assert len(dfs) >= 1, "At least one DataFrame is required"
    base = dfs[0]
    for df in dfs[1:]:
        base = base.join(df, on=on, how=how, suffix="_r")
        if coalesce:
            # aynı isimli sütunlar için _r ile gelenleri birleştir
            for c in base.columns:
                if c.endswith("_r"):
                    orig = c[:-2]
                    if orig in base.columns:
                        base = base.with_columns(pl.coalesce([pl.col(orig), pl.col(c)]).alias(orig)).drop(c)
    return base
