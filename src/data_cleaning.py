import pandas as pd

def clean_data(df):
    df = df.copy()

    # Remove completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate rows
    df = df.drop_duplicates()

    return df