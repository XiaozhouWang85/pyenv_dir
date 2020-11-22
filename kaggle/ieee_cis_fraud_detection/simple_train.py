import pickle

from imblearn.over_sampling import SMOTE

import numpy as np
import pandas as pd

from src import data_transformations as dt
from src import models as md

SEED = 123

# define continuous and categorical variables
cont_vars = ["TransactionAmt"] + \
["C"+str(i+1) for i in range(14)] + \
["D"+str(i+1) for i in range(15)] + \
["V"+str(i+1) for i in range(339)] + \
["dist"+str(i+1) for i in range(2)]

cat_vars = ['ProductCD', 'addr1', 'addr2', 'P_emaildomain', 'R_emaildomain',"Time of Day"] + \
["card"+str(i+1) for i in range(6)] + \
["M"+str(i+1) for i in range(9)] 

# generate time of day

lgb_params = {
  'objective':'binary',
  'boosting_type':'gbdt',
  'metric':'auc',
  'n_jobs':-1,
  'learning_rate':0.02,
  'num_leaves': 2**8,
  'max_depth':-1,
  'tree_learner':'serial',
  'colsample_bytree': 0.7,
  'subsample_freq':1,
  'subsample':0.7,
  'n_estimators':10000,
  'max_bin':255,
  'verbose':-1,
  'seed': SEED,
  'early_stopping_rounds':100, 
}

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

categories = load_obj('categories' )
nn_params = {
  'categories_dict':categories, 
  'layers': [200,100], 
  'epochs': 10,
}

print("Read in data")
train = pd.read_csv("data/train_transaction.csv",nrows=10000)
holdout = pd.read_csv("data/test_transaction.csv",nrows=10000)

y = train["isFraud"].copy()
IDs = holdout["TransactionID"].copy()
train.drop("isFraud",axis=1,inplace=True)

pipe = dt.full_pipeline(cont_vars,cat_vars)
train, holdout = pipe.training_transforms(train,holdout)

#estimators, metrics = md.train_kfolds(train, y, md.train_lgb, cat_vars, SEED, 5, lgb_params = lgb_params)
#estimators, metrics = md.train_single(train, y, md.train_lgb, cat_vars, SEED, 0.33, lgb_params = lgb_params)

estimators, metrics = md.train_single(train, y, md.train_tabular_nn, cat_vars, SEED, 0.33, tab_nn_params = nn_params)

predictions = np.zeros(holdout.shape[0])

for est in estimators:
    y_pred = est.predict(holdout)
    predictions += y_pred/5

pd.DataFrame({'TransactionID': IDs, 'isFraud': predictions},columns=["TransactionID","isFraud"]).to_csv("submission.csv",index=False)
