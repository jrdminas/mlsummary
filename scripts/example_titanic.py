import numpy as np
import pandas as pd
from mlsummary import association_matrix
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


np.random.seed(0)


def recode(x: pd.DataFrame) -> pd.DataFrame:
    """
    Esta función tranforma las columnas del dataframe de entrada.

    :param x: DataFrame con las columnas a transformar
    :return: DataFrame de salida con las columnas transformadas
    """
    __cols = x.columns
    __x = x.copy()
    if 'SibSp' in __cols:
        __x['SibSp'] = __x['SibSp'].fillna(0).map(lambda v: v if v < 4 else 4)
        # Se imputan los nulos a cero
    if 'Parch' in __cols:
        __x['Parch'] = __x['Parch'].fillna(0).map(lambda v: v if v < 3 else 3)
        # Se imputan los nulos a cero
    if 'Fare' in __cols:
        __cut_labels = ['[0,8]', '(8,20]', '(20,60]', '(60,100]', '(100,Inf]']
        __cut_bins = [0, 8, 20, 60, 100, np.Inf]
        __x['Fare'] = pd.cut(__x['Fare'].fillna(15), bins=__cut_bins, 
                             labels=__cut_labels, include_lowest=True)
        # Se imputan los nulos en el segundo grupo: 8 - 20
    if 'Age' in __cols:
        __cut_labels = ['[0,6]', '(6,15]', '(15,25]', '(25,30]', '(30,40]', '(40,50]', '(50,100]']
        __cut_bins = [0, 6, 15, 25, 30, 40, 50, 100]
        __x['Age'] = pd.cut(__x['Age'].fillna(99), bins=__cut_bins, 
                            labels=__cut_labels, include_lowest=True)
        # Se imputan los nulos en el último grupo: 50 - 100
    if 'Embarked' in __cols:
        __x['Embarked'] = __x['Embarked'].fillna('S')
    
    if 'Sex' in __cols:
        __x['Sex'] = __x['Sex'].fillna('male')
    if 'Pclass' in __cols:
        __x['Pclass'] = __x['Pclass'].fillna(3)

    return __x


if __name__=='__main__':
    
    selected_cols = ['Pclass','Sex','Embarked','SibSp','Parch','Age','Fare']
    df_train = pd.read_csv('data/train.csv')
    df_train = recode(x=df_train)
    df_train = df_train[selected_cols]
    print(df_train.head(5))
    print(df_train.shape)

    tic = datetime.now()
    print(pd.DataFrame(association_matrix(x=df_train.values), columns=selected_cols, index=selected_cols))
    print(f"association-matrix time: {datetime.now() - tic}")
