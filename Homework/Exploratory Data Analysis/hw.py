import pandas as pd
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt

baseball = pd.read_csv("Baseball_salary.csv", header=None)
baseball.drop(baseball.index[0], inplace=True)
baseball[[1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19]] = \
    baseball[[1,2,3,4,5,6,7,8,9,10,11,12,13,16,17,18,19]].astype(float)

baseball['output'] = np.log(baseball[19])

print(baseball)
print(baseball.describe(include='all'))

baseball.hist()
plt.show()