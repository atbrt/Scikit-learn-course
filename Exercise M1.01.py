
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

n_samples_to_plot = 5000
columns = ["Culmen Length (mm)", "Culmen Depth (mm)"]
target_column="Species"
_ = sns.pairplot(
    data=penguins[:n_samples_to_plot],
    vars=columns,
    hue=target_column,
    plot_kws={"alpha": 0.2},
    height=3,
    diag_kind="hist",
    diag_kws={"bins": 30},
)
plt.show()
