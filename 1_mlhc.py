from utils.etl.fetch_data import load_housing_data
from utils.etl.sklearn_extender import CombinedAttributesAdder

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

import pandas as pd
import matplotlib.pyplot as plt

# 1) Get data
housing_data = load_housing_data()

# 2) Explore the data
print(housing_data.info())
print(housing_data.describe())

# Histogram
housing_data.hist(bins=50, figsize=(20, 15))

# Geographic plot
housing_data.plot(
    kind='scatter',
    x='longitude',
    y='latitude',
    alpha=0.4,
    s=housing_data['population']/100,
    label="Population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    sharex=False
)

# Correlation of features to y variable
corr_matrix = housing_data.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# 3) Prepare the data
# a) Data Cleaning
train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

df = train_set.drop(columns=['median_house_value'])
df_numerical = df.drop(columns=['ocean_proximity'])

df_numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])

num_attribs = list(df_numerical)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([("num", df_numerical_pipeline, num_attribs), ("cat", OneHotEncoder(), cat_attribs)])

# 4) Shortlist promising models

# 5) Fine-tune the system

# 6) Present solution

# 7) Launch model
