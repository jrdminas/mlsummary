import numpy as np
import polars as pl
from mlsummary import association_matrix
from datetime import datetime
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer


np.random.seed(0)
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(100)


if __name__=='__main__':
    N = 1000
    xcat1 = np.random.choice(a=["Porsche","Jaguar", "Mercedes", "Kia"], size=N, p=[0.10, 0.15, 0.25, 0.5] )
    xcat2 = np.random.choice(a=["Porsche","Jaguar", "Mercedes", "Kia"], size=N, p=[0.15, 0.10, 0.25, 0.5] )
    xx1 = np.column_stack([xcat1, xcat2])
    print(xx1.shape)

    tic = datetime.now()
    print(association_matrix(x=xx1))
    print(f"association-matrix time: {datetime.now() - tic}")

