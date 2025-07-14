import pandas as pd

def load_required_attributes_from_raw(path: str, include_labels: bool = True):
    """
    Loads only the columns required for TitanicInputTransformer from the raw Titanic CSV.
    If include_labels is True, returns (X, y). Otherwise, returns X only.
    """
    required_cols = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
    if include_labels:
        required_cols = ['Survived'] + required_cols
    df = pd.read_csv(path, usecols=[col for col in required_cols if col in pd.read_csv(path, nrows=0).columns])
    if 'PassengerId' in df.columns:
        df = df.set_index('PassengerId')
    if include_labels:
        X = df.drop('Survived', axis=1)
        y = df['Survived']
        return X, y
    else:
        return df
