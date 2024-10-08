import pandas as pd
from sklearn.neighbors import KNeighborsClassifier


adult_census = pd.read_csv("bases de données/adult-census-numeric.csv")
adult_census_test=pd.read_csv("bases de données/adult-census-numeric-test.csv")
data = adult_census.drop(columns="class")
target = adult_census["class"]
data_test = adult_census_test.drop(columns="class")
target_test = adult_census_test["class"]

model=KNeighborsClassifier(n_neighbors=50)
_=model.fit(data, target)

target_predicted = model.predict(data)

accuracy = model.score(data, target)
model_name = model.__class__.__name__

print(f"The test accuracy using a {model_name} is {accuracy:.3f}")


accuracy2 = model.score(data_test, target_test)


print(f"The test accuracy using a {model_name} is {accuracy2:.3f}")
