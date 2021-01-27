import pandas as pd
import numpy as np

df = pd.read_excel("./excel-comp-data.xlsx")
print(df.drop("name", axis=1))
print(df.head(2).T)

print(df[["account", "street", "state"]].head())
account_series = df["account"]

print(df[account_series > 200000])

df.index = df["account"]
print(df[["name", "street"]].iloc[:2])

matrix = df.values
print(matrix[:, -3:].sum(axis=1))

s = pd.Series(np.nan, index=range(10, 0, -1))
print(s.loc[:3])
print(s.iloc[:3])

print(df["account"][df["account"] < 200000])