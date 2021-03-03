from utils.etl.fetch_data import load_housing_data

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

# 1) Get data
housing_data = load_housing_data()

train_set, test_set = train_test_split(housing_data, test_size=0.2, random_state=42)

df = train_set

# 2) Explore the data
# Histogram
df.hist(bins=50, figsize=(20, 15))

# Geographic plot
df.plot(
    kind='scatter',
    x='longitude',
    y='latitude',
    alpha=0.4,
    s=df['population']/100,
    label="Population",
    figsize=(10, 7),
    c="median_house_value",
    cmap=plt.get_cmap("jet"),
    colorbar=True,
    sharex=False
)

# Correlation of features to y variable
corr_matrix = df.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)

# 3) Prepare the data

# 4) Shortlist promising models

# 5) Fine-tune the system

# 6) Present solution

# 7) Launch model
