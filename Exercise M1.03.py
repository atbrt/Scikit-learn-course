import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

adult_census = pd.read_csv("bases de données/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=target_name)

numerical_columns = ["age", "capital-gain", "capital-loss", "hours-per-week"]

data_numeric = data[numerical_columns]

data_train, data_test, target_train, target_test = train_test_split(
    data_numeric, target, random_state=42, test_size=0.25
)

model=DummyClassifier(strategy="constant", constant=' >50K')
model.fit(data_train, target_train)
accuracy=model.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy:.3f}")

model2=DummyClassifier(strategy="constant", constant=' <=50K')
model2.fit(data_train, target_train)
accuracy2=model2.score(data_test, target_test)
print(f"Accuracy of logistic regression: {accuracy2:.3f}")

