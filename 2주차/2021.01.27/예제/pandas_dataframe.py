from pandas import Series, DataFrame
import pandas as pd
import numpy as np

raw_data = {
    "first_name": ["Jason", "Molly", "Tina", "Jake", "Amy"],
    "last_name": ["Miller", "Jacobson", "Ali", "Milner", "Cooze"],
    "age": [42, 52, 36, 24, 73],
    "city": ["San Francisco", "Baltimore", "Miami", "Douglas", "Boston"],
}

df1 = pd.DataFrame(raw_data, columns=["first_name", "last_name", "age", "city", "debt"],
                             index=["a", "b", "c", "d", "e"])

df2 = pd.DataFrame(raw_data, columns=["first_name", "age"])
print("print data frame")
print(df1)
print(df2)

print("\nprint data frame column")
print(df1.first_name)
print(df1["first_name"])

print("\ndata frame indexing")
print(df2.first_name.loc[:3])
print(df2.first_name.iloc[:3])

print("\nnew data")
df1.debt = df1.age > 40
print(df1)

print("\ndelete data")
print(df1.drop("debt", axis=1))
del df1["debt"]
print(df1)

print("\nselection and drop")
print(df1[["first_name", "age", "last_name"]])


print(df1[["first_name", "age"]][df1.age < 50])
