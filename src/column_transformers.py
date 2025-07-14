from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class DropSpareDummies(BaseEstimator, TransformerMixin):
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        # No fitting necessary, but method required by scikit-learn
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop)


import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicInputTransformer(BaseEstimator, TransformerMixin):
    """
    Transforms Titanic input DataFrame into model-ready features.
    Always receives and returns a DataFrame.
    """
    def __init__(self, columns_to_drop=None):
        self.pclass_map = {'1st Class': 1, '2nd Class': 2, '3rd Class': 3, 1: 1, 2: 2, 3: 3}
        self.sex_map = {'male': 'male', 'female': 'female', 'Male': 'male', 'Female': 'female'}
        self.embarked_map = {'C': 'C', 'Q': 'Q', 'S': 'S'}
        self.dummy_columns = ['C', 'Q', 'S', 'female', 'male', 'Class_1', 'Class_2', 'Class_3']
        self.columns_to_drop = columns_to_drop or ['C', 'female', 'Class_1']
    def fit(self, X, y=None):
        return self

    def transform(self, df):
        required_cols = ['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Embarked']
        missing = [col for col in required_cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in input: {missing}")
        # Ensure input is a DataFrame
        df = df.copy()
        df['Pclass'] = df['Pclass'].map(self.pclass_map)
        df['Sex'] = df['Sex'].map(self.sex_map)
        df['Embarked'] = df['Embarked'].map(self.embarked_map)
        df['Age'] = df['Age'].fillna(28)
        df['Fare'] = df['Fare'].fillna(32)
        df['Embarked'] = df['Embarked'].fillna('S')

        # One-hot encode
        embark_dummies = pd.get_dummies(df['Embarked'])
        sex_dummies = pd.get_dummies(df['Sex'])
        pclass_dummies = pd.get_dummies(df['Pclass'], prefix="Class")

        # Concatenate all features
        out = pd.concat([
            df[['Age', 'SibSp', 'Parch', 'Fare']].reset_index(drop=True),
            embark_dummies.reindex(columns=['C', 'Q', 'S'], fill_value=0).reset_index(drop=True),
            sex_dummies.reindex(columns=['female', 'male'], fill_value=0).reset_index(drop=True),
            pclass_dummies.reindex(columns=['Class_1', 'Class_2', 'Class_3'], fill_value=0).reset_index(drop=True)
        ], axis=1)

        # Drop spare dummies if needed
        if self.columns_to_drop:
            out = out.drop(columns=self.columns_to_drop, errors='ignore')

        return out