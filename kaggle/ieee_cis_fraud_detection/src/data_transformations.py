import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.base import BaseEstimator, TransformerMixin


class continuos_transform(BaseEstimator, TransformerMixin):

  def __init__(self,cont_vars):
    self.cont_vars = cont_vars
    self.scaler = {var: StandardScaler() for var in cont_vars}

  def fit(self, X, y=None):
    for var in self.cont_vars:
      self.scaler[var].fit(X[var].values.reshape(-1,1))

    return self

  def transform(self, X):

    for var in self.cont_vars:
      X[var] = self.scaler[var].transform(X[var].values.reshape(-1,1))

    return X

class categorical_transform(BaseEstimator, TransformerMixin):

  def __init__(self,cat_vars):
    self.cat_vars = cat_vars
    self.le = {var: LabelEncoder() for var in cat_vars}
    self.idx_list = {}

  def fit(self, X, y=None):
    for var in self.cat_vars:
      self.idx_list[var] = X[var].value_counts()[X[var].value_counts()<=100].index.tolist()
      X.loc[X[var].isin(self.idx_list[var]),var] = "Others"
      self.le[var].fit(X[var])

    return self

  def transform(self, X):
    for var in self.cat_vars:
      X.loc[X[var].isin(self.idx_list[var]),var] = "Others"
      X[var] = self.le[var].transform(X[var])

    return X

class full_pipeline():

  def __init__(self,cont_vars,cat_vars):
    self.cont_vars = cont_vars
    self.cat_vars = cat_vars

  def training_transforms(self,train,test):
    train = train.copy()
    test = test.copy()

    train = self.preprocessing(train)
    test = self.preprocessing(test)

    self.cont_pipe = continuos_transform(self.cont_vars)
    self.cat_pipe = categorical_transform(self.cat_vars)

    self.cat_pipe.fit(
      pd.concat([train,test],axis=0)
    )

    train = self.cont_pipe.fit_transform(
      self.cat_pipe.transform(train)
    )

    test = self.cont_pipe.fit_transform(
      self.cat_pipe.transform(test)
    )

    return train[self.cont_vars + self.cat_vars], test[self.cont_vars + self.cat_vars]
  
  def predict_transforms(self,test):
    test = test.copy()

    test = self.preprocessing(test)

    test = self.cont_pipe.fit_transform(
      self.cat_pipe.transform(test)
    )

    return test[cont_vars + cat_vars]

  def preprocessing(self, X):
    X["Time of Day"] = np.floor(X["TransactionDT"]/3600/183)
    X.drop("TransactionDT",axis=1,inplace=True)
    X[self.cat_vars] = X[self.cat_vars].astype('str')
    X[self.cat_vars] = X[self.cat_vars].fillna("Missing")
    return X