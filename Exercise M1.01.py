
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
penguins=pd.read_csv("bases de données/penguins_classification.csv")



penguins.head()

especes=penguins["Species"]
especes.value_counts()
penguinsnum=penguins.select_dtypes(include=['number'])
_=penguinsnum.hist(figsize=(20, 14))
plt.show()

