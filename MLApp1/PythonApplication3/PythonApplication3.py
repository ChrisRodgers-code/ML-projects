import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import tarfile
from six.moves import urllib
from sklearn.model_selection import StratifiedShuffleSplit
from zlib import crc32
from pandas.plotting import scatter_matrix

np.random.seed(42)

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path,"housing.csv")
    return pd.read_csv(csv_path)

def split_train_test (data, test_ratio):
    shuffled_indicies = np.random.permutation(len(data))
    test_set_size = int(len(data)*test_ratio)
    test_indicies = shuffled_indicies[:test_set_size]
    train_indicies = shuffled_indicies[test_set_size:]
    return data.iloc[train_indicies], data.iloc[test_indicies]

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data,test_ratio,id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_,test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]

housing = load_housing_data()
#print(housing.head())
#print(housing.info())

#print(housing["ocean_proximity"].value_counts())
#print(housing.describe())

#housing.hist(bins=50, figsize = (20,15))
#save_fig("attribute_histogram_plots")
#plt.show()

train_set, test_set = split_train_test(housing, 0.2)
#print(len(train_set), "train +", len(test_set), "test")

#adding labels
housing_with_id = housing.reset_index()
train_set, test_set = split_train_test_by_id(housing_with_id,0.2,"index")
#housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
#print(test_set.head())

#housing["median_income"].hist()
#plt.show()

#housing["income_cat"] = pd.cut(housing["median_income"],
#                               bins=[0.,1.5,3.0,4.5,6.,np.inf],
#                               labels=[1,2,3,4,5])
#print(housing["income_cat"].value_counts())

#housing["income_cat"].hist()
#plt.show()

housing["income_cat"] = np.ceil(housing["median_income"]/1.5)
housing["income_cat"].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing,housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#print(strat_test_set["income_cat"].value_counts()/len(strat_test_set))

for set in (strat_train_set, strat_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

#Discover and visualize the data to gain insights
housing = strat_train_set.copy()
housing.plot.scatter(figsize=(10,7),x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", c='median_house_value',
             cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
plt.show()

#Looking For Correlations
corr_matrix = housing.corr()
#print(corr_matrix["median_house_value"].sort_values(ascending=False))

#attributes = ["median_house_value", "median_income", "total_rooms",
#              "housing_median_age"]
#scatter_matrix(housing[attributes],figsize=(12,8))
#plt.show()

housing.plot.scatter(x="median_income", y="median_house_value", alpha=0.1)
plt.axis([0,16,0,550000])
plt.show()