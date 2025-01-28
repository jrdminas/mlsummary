import numpy as np
import polars as pl
from mlsummary import vif
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
    xx = np.random.normal(0.01, 1, (N, 10))
    xcat = np.random.choice(a=["Porsche","Jaguar", "Mercedes", "Kia"], size=N, p=[0.10, 0.15, 0.25, 0.5] )
    xx1 = np.column_stack([xx[:, :5], 5. * xx[:,(2,3)], xx[:, 5:], xcat])
    print(xx1.shape)

    num_transformer = Pipeline(
            steps=[("imputer", SimpleImputer(strategy="median")), 
                   ("scaler", StandardScaler())]
            )
    cat_transformer = Pipeline(steps=[("encoder", OrdinalEncoder())])
    preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_transformer, list(range(12))),
                ("cat", cat_transformer, [12])]
            )
    xx1 = preprocessor.fit_transform(xx1)


    print(pl.DataFrame(np.corrcoef(xx1, rowvar=False)))

    tic = datetime.now()
    print(pl.from_dict(vif(xx1, method='inv')))
    print(f"vif[inv] time: {datetime.now() - tic}")

    tic = datetime.now()
    print(pl.from_dict(vif(xx1)))
    print(f"vif[linregress] time: {datetime.now() - tic}")

