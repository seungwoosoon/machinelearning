import numpy as np
import pandas as pd
arr = np.array([2,3,4])
print(arr)
df = pd.DataFrame([[4,5,6],
        [4,5,6],
        [4,5,6]],
        index = [1,2,3], columns=["a","b","c"])

print(df)