# D. Akman
# GPL-3.0, 2022

import pandas as pd
import os
import io
import requests
import ssl
from sklearn import preprocessing
from sklearn.utils import shuffle


def prepare_dataset_for_modeling(dataset_name,
                                 pred_type,
                                 data_directory=None,
                                 na_values='?',
                                 n_samples_max=None,
                                 random_state=999,
                                 drop_const_columns=True,
                                 scale_data=True):
    """
    ASSUMPTION 1: The target variable is the LAST column in the dataset.
    ASSUMPTION 2: First row in the file is the header row.
    :param dataset_name: name of the dataset - will be passed to pd.read_csv()
    :param pred_type: if 'c' for classification, y is assumed categorical and it will be label-encoded for model fitting
                      if 'r' for regression, y is assumed numerical
    :param data_directory: directory of the dataset. If None, the dataset will be read in from GitHub
    :param na_values: Additional strings to recognize as NA/NaN - will be passed to pd.read_csv()
    :param n_samples_max: max no. of instances to sample (if not None)
    :param random_state: seed for shuffling (and sampling) instances
    :param drop_const_columns: if True, drop constant-value columns (*after* any sampling)
    :param scale_data: whether the descriptive features (and y also if regression) are to be min-max scaled
    :return: x and y NumPy arrays ready for model fitting
    """

    if pred_type not in ['c', 'r']:
        raise ValueError("Prediction type needs to be either 'c' for classification or 'r' for regression.")

    if data_directory:
        # read in from local directory
        df = pd.read_csv(data_directory + dataset_name, na_values=na_values, header=0)
    else:
        # read in the data file from GitHub into a Pandas data frame
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
                getattr(ssl, '_create_unverified_context', None)):
            ssl._create_default_https_context = ssl._create_unverified_context
        github_location = 'https://raw.githubusercontent.com/vaksakalli/datasets/master/'
        dataset_url = github_location + dataset_name.lower()
        df = pd.read_csv(io.StringIO(requests.get(dataset_url).content.decode('utf-8')), na_values=na_values, header=0)

    # drop missing values before (any) sampling
    df = df.dropna()

    # shuffle dataset in case of a pattern and also subsample if requested
    # but do not sample more than the available number of observations (*after* dropping missing values)
    # n_samples_max = None results in no sampling; just shuffling
    n_observations = df.shape[0]  # no. of observations in the dataset
    n_samples = n_observations  # initialization - no. of observations after (any) sampling
    if n_samples_max and (n_samples_max < n_observations):
        # do not sample more rows than what is in the dataset
        n_samples = n_samples_max
    df = shuffle(df, n_samples=n_samples, random_state=random_state)

    if drop_const_columns:
        # drop constant columns (after sampling)
        df = df.loc[:, df.nunique() > 1]

    # drop any duplicate rows
    df = df.drop_duplicates(ignore_index=True)

    # last column is y (target feature)
    y = df.iloc[:, -1].values
    # everything else is x (set of descriptive features)
    x = df.iloc[:, :-1]

    # get all columns that are objects
    # these are assumed to be nominal categorical
    categorical_cols = x.columns[x.dtypes == object].tolist()

    # if a categorical feature has only 2 levels:
    # encode it as a single binary variable
    for col in categorical_cols:
        n = len(x[col].unique())
        if n == 2:
            x[col] = pd.get_dummies(x[col], drop_first=True)

    # for categorical features with >2 levels: use one-hot-encoding
    # below, numerical columns will be untouched
    x = pd.get_dummies(x).values

    if scale_data:
        # scale x between 0 and 1
        x = preprocessing.MinMaxScaler().fit_transform(x)
        if pred_type == 'r':
            # also scale y between 0 and 1 for regression problems
            y = preprocessing.MinMaxScaler().fit_transform(y.reshape(-1, 1)).flatten()

    if pred_type == 'c':
        # label-encode y for classification problems
        y = preprocessing.LabelEncoder().fit_transform(y)

    return x, y


# # example: how to run this script
# x, y = prepare_dataset_for_modeling('sonar.csv', pred_type='c')
