# datasets

A collection of public datasets for machine learning research & teaching.

Automated script to prepare these datasets for predictive modeling:
https://github.com/akmand/datasets/blob/main/prepare_dataset_for_modeling_github.py

Assumptions:
- First row is the header row.
- Target feature is the last column.
- There might be missing values.
- Categorical features are encoded as strings - so that pd.get_dummies() work correctly.
