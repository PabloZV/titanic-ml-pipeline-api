import pandas as pd
import numpy as np
from typing import Tuple

def load_and_preprocess_data(train_path: str) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """
    Loads Titanic data, applies preprocessing, and returns features and labels.
    This function does NOT split the data.
    """
    # Load data
    df = pd.read_csv(train_path)

    # Drop unnecessary columns
    drop_cols = ['Name', 'Ticket', 'Cabin']
    df = df.drop(columns=drop_cols)

    # Fill missing values
    df['Age'] = df['Age'].fillna(df['Age'].median())
    df['Embarked'] = df['Embarked'].fillna('S')

    # One-hot encoding
    embark_dummies = pd.get_dummies(df['Embarked'])
    sex_dummies = pd.get_dummies(df['Sex'])
    pclass_dummies = pd.get_dummies(df['Pclass'], prefix="Class")
    df = df.drop(['Embarked', 'Sex', 'Pclass'], axis=1)
    df = df.join([embark_dummies, sex_dummies, pclass_dummies])

    # Separate features and labels
    X = df.drop('Survived', axis=1)
    y = df['Survived']
    return X, y
