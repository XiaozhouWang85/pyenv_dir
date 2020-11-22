import lightgbm as lgb
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold, train_test_split
import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Dropout, BatchNormalization
from tensorflow.keras import Model

def train_lgb(X_train, X_test, y_train, y_test, cat_vars, lgb_params):
  train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_vars)  
  test_data = lgb.Dataset(X_test, label=y_test, categorical_feature=cat_vars)
  lgb_params = lgb_params.copy()


  #Extra step here prevents warnings being thrown
  if 'n_estimators' in lgb_params:
    num_boost_round = lgb_params["n_estimators"]
    del lgb_params['n_estimators']
  else:
    num_boost_round = 1000
  
  if 'early_stopping_rounds' in lgb_params:
    early_stopping = lgb_params["early_stopping_rounds"]
    del lgb_params['early_stopping_rounds']
  else:
    early_stopping = 100

  estimator = lgb.train(
    lgb_params,
    train_data,
    num_boost_round = num_boost_round,
    early_stopping_rounds = early_stopping,
    valid_sets = [train_data, test_data],
    verbose_eval = 200
  )   
  return estimator


def train_tabular_nn(X_train, X_test, y_train, y_test, cat_vars, tab_nn_params):
  cont_vars = [var for var in X_train.columns if var not in cat_vars]
  estimator = tabular_nn(cat_vars, cont_vars, **tab_nn_params)
  estimator.fit(X_train,y_train)

  return estimator


class tabular_nn():

  def __init__(self, cat_vars,cont_vars, categories_dict, layers, epochs):
    self.cat_vars = cat_vars
    self.cont_vars = cont_vars
    self.epochs = epochs

    self.model = self.combined_network(cat_vars,categories_dict,cont_vars, layers)
    opt = tf.keras.optimizers.Adam(0.0001)
    self.model.compile(optimizer=opt,loss='binary_crossentropy',metrics=["accuracy"])

  def calculate_embedding_size(self, cat_col,categories_dict):
    num_classes = len(categories_dict[cat_col])
    return int(min(600,round(1.6*num_classes**0.56)))

  def reformat_dataframe(self, df):
    data_list = []
    for var in self.cat_vars:
      data_list.append(df[var].values)

    data_list.append(df[self.cont_vars].values)
    return data_list

  def fit(self, X,y):
    data_list = self.reformat_dataframe(X)
    self.model.fit(data_list,y,epochs=self.epochs)

  def predict(self, X):
    data_list = self.reformat_dataframe(X)
    self.model.predict(data_list)

  def combined_network(self, cat_vars,categories_dict,cont_vars, layers):
    input_layers = []
    embedding_layers = []
    
    cat_input_layers ={}

    # create embedding layer for each categorical variables
    for var in cat_vars:

      emb_sz = self.calculate_embedding_size(var,categories_dict)
      vocab = len(categories_dict[var]) +1

      cat_input = Input(shape=(1,))
      cat_embedding = Embedding(vocab,emb_sz,input_length=1)(cat_input)
      cat_embedding_resized = Reshape(target_shape=(emb_sz,))(cat_embedding)
      
      input_layers.append(cat_input)
      embedding_layers.append(cat_embedding_resized)
    
    # concat continuous variables with embedded variables
    cont_input = Input(shape=(len(cont_vars),))
    cont_input_norm = BatchNormalization()(cont_input)

    input_layers.append(cont_input)
    embedding_layers.append(cont_input_norm)

    x = Concatenate()(embedding_layers)
    
    # add user-defined fully-connected layers separated with batchnorm and dropout layers
    for i in range(len(layers)):
      if i ==0:
        x = Dense(layers[i],activation="relu")(x)
      else:
        x = BatchNormalization()(x)
        x = Dropout(0.5)(x)
        x = Dense(layers[i],activation="relu")(x)
    output = Dense(1,activation="sigmoid")(x)
    model = Model(input_layers,output)
    return model

def train_kfolds(train_df, labels, model_fn, cat_vars, random_state, n_splits, **kwargs):

  metrics = []
  estimators = []

  kf = KFold(n_splits=n_splits,shuffle=True, random_state=random_state)
  print("Starting Kfolds training")
  for i, indices in enumerate(kf.split(train_df)):
    print("Processing split {}".format(i+1))
    train_index, test_index = indices
    X_train, X_test = train_df.iloc[train_index], train_df.iloc[test_index]
    y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
    
    print("Training model for split {}".format(i+1))

    estimator = model_fn(X_train, X_test, y_train, y_test, cat_vars, **kwargs)

    y_pred = estimator.predict(X_test)

    metrics.append(roc_auc_score(y_test,y_pred))
    estimators.append(estimator)

  print ('Average metrics =',np.mean(metrics))
  return estimators, metrics

def train_single(train_df,labels, model_fn, cat_vars, random_state,test_size, **kwargs):

  X_train, X_test, y_train, y_test = train_test_split(
    train_df, labels, test_size=test_size, random_state=random_state
  )

  estimator = model_fn(X_train, X_test, y_train, y_test, cat_vars, **kwargs)

  y_pred = estimator.predict(X_test)

  metrics = roc_auc_score(y_test,y_pred)

  
  return [estimator], [metrics]
