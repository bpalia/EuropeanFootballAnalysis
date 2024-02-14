# Last updated October 10, 2023
# Version 0.1.0

import pandas as pd
import sqlite3


def optimize_dtypes(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Function to optimize data types. Assumes no NaNs are in the dataframe."""
    in_memory_usage = df.memory_usage(deep=True).sum()
    cols = df.columns
    if "date" in cols:
        df["date"] = df["date"].astype("datetime64[D]")
    num_cols = df.select_dtypes("number").columns
    df[num_cols] = df[num_cols].apply(pd.to_numeric, downcast="unsigned")
    obj_cols = df.select_dtypes("object").columns
    df[obj_cols] = df[obj_cols].astype("category")
    out_memory_usage = df.memory_usage(deep=True).sum()
    if verbose:
        print(
            f"Compression ratio: {round(in_memory_usage/out_memory_usage, 2)}."
        )
    return df


def drop_rows_nans(df: pd.DataFrame) -> pd.DataFrame:
    """Function to print percentage of rows with at least one NaN."""
    na_prop = df.isna().any(axis=1).sum() / df.shape[0] * 100
    print(f"Dropped rows due to NaNs: {round(na_prop, 2)}%.")
    df.dropna(axis=0, how="any", inplace=True)
    return df


def select_columns(
    table: str, expression: str, conn: sqlite3.Connection
) -> list[str]:
    """Function to select and print specific columns from sqlite table based
    on regex."""
    query = """
        SELECT name 
          FROM pragma_table_info(?)
        """
    cols = pd.read_sql(sql=query, con=conn, params=(table,))["name"]
    cols = cols[cols.str.contains(expression, regex=True)].to_list()
    print(", ".join(cols))
    return cols
