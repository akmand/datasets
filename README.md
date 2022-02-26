# datasets
Script to prepare datasets for modeling:

Assumptions:
- First row is the header row.
- Target feature is the last column.
- There might be missing values.
- Categorical features are encoded as strings - so that pd.get_dummies() work correctly.
