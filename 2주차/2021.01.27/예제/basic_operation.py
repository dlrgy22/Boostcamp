import pandas as pd
from pandas import Series
from pandas import DataFrame

import numpy as np
s1 = Series(range(1, 6), index=list("abced"))
s2 = Series(range(5, 11), index=list("bcedef"))
print(f"{s1}\n{s2}")
print(s1 + s2)
print(s1.add(s2, fill_value=0))
print(s1)

df1 = DataFrame(np.arange(9).reshape(3, 3), columns=list("abc"))
df2 = DataFrame(np.arange(16).reshape(4, 4), columns=list("abcd"))
print(df1 + df2)
print(df1.add(df2, fill_value=0))


s = Series(np.arange(10, 14), index=list("abcd"))
s2 = Series(np.arange(10, 14))
print(df2 + s2)
print(df2)
print(df2.add(s2, axis=0))
