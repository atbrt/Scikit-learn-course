import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.compose import make_column_selector as selector


adult_census = pd.read_csv("bases de données/adult-census.csv")

target_name = "class"
target = adult_census[target_name]
data = adult_census.drop(columns=[target_name, "education-num"]) #We drop the variable that is two times present in the data

categorical_columns_selector = selector(dtype_include=object)
categorical_columns = categorical_columns_selector(data)
data_categorical = data[categorical_columns]

model = make_pipeline(
    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), 
    LogisticRegression(max_iter=500)
)

cv_results = cross_validate(model, data_categorical, target)

cv_results

scores = cv_results["test_score"]
print(f"The accuracy is: {scores.mean():.3f} ± {scores.std():.3f}")

model2 = make_pipeline(
    OneHotEncoder(handle_unknown="ignore"), 
    LogisticRegression(max_iter=500)
)

cv_results2 = cross_validate(model2, data_categorical, target)

cv_results2

scores2 = cv_results2["test_score"]
print(f"The accuracy is: {scores2.mean():.3f} ± {scores2.std():.3f}")
