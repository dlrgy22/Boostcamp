import pandas as pd
from pandas import Series

values = [1, 2, 3, 4]
index = ["A", "B", "C", "D"]

example_series1 = Series(values, index)
print(example_series1)

dict_data = {"A":1, "B":2, "C":3, "D":4,}
example_series2 = Series(dict_data, dtype=float)
print(example_series2)

print(f"\nexample_series2 index : {example_series2.index}")
print(f"example_series2 values : {example_series2.values}")

dict_data = {"A":1, "B":2, "C":3, "D":4,}
indexes = ["A", "B", "C", "D", "E"]
example_series3 = Series(dict_data, index=indexes)
print(example_series3)





