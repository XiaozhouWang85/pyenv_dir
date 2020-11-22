
import pickle
import pandas as pd

import tensorflow as tf
from tensorflow.keras.layers import Input, Embedding, Dense, Reshape, Concatenate, Dropout, BatchNormalization
from tensorflow.keras import Model

def load_obj(name ):
    with open('data/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)

categories = load_obj('categories' )

x_train = pd.read_csv("data/x_train.csv")
x_test = pd.read_csv("data/x_test.csv")
y_train = pd.read_csv("data/y_train.csv")

cont_vars = ["TransactionAmt"]
cat_vars = ["ProductCD","addr1","addr2","P_emaildomain","R_emaildomain","Time of Day"] + [col for col in x_train.columns if "card" in col]

# get embedding size for each categorical variable
def get_emb_sz(cat_col,categories_dict):
	num_classes = len(categories_dict[cat_col])
	return int(min(600,round(1.6*num_classes**0.56)))

# define the neural networks

def combined_network(cat_vars,categories_dict,cont_vars, layers):
	inputs = []
	embeddings = []
	emb_dict ={}
	# create embedding layer for each categorical variables
	for i in range(len(cat_vars)):
		emb_dict[cat_vars[i]] = Input(shape=(1,))
		emb_sz = get_emb_sz(cat_vars[i],categories_dict)
		vocab = len(categories_dict[cat_vars[i]]) +1
		embedding = Embedding(vocab,emb_sz,input_length=1)(emb_dict[cat_vars[i]])
		embedding = Reshape(target_shape=(emb_sz,))(embedding)
		inputs.append(emb_dict[cat_vars[i]])
		embeddings.append(embedding)
	
	# concat continuous variables with embedded variables
	cont_input = Input(shape=(len(cont_vars),))
	embedding = BatchNormalization()(cont_input)

	inputs.append(cont_input)
	embeddings.append(embedding)
	x = Concatenate()(embeddings)
	
	# add user-defined fully-connected layers separated with batchnorm and dropout layers
	for i in range(len(layers)):
		if i ==0:
			x = Dense(layers[i],activation="relu")(x)
		else:
			x = BatchNormalization()(x)
			x = Dropout(0.5)(x)
			x = Dense(layers[i],activation="relu")(x)
	output = Dense(1,activation="sigmoid")(x)
	model = Model(inputs,output)
	return model

layers = [200,100]
model = combined_network(cat_vars,categories,cont_vars, layers)
opt = tf.keras.optimizers.Adam(0.0001)
model.compile(optimizer=opt,loss='binary_crossentropy',metrics=["accuracy"])

# process x_train input to fit model
input_list = []
for i in cat_vars:
	input_list.append(x_train[i].values)

input_list.append(x_train[cont_vars].values)

# modify x_test input to fit model
test_list = []
for i in cat_vars:
	test_list.append(x_test[i].values)

test_list.append(x_test[cont_vars].values)

model.fit(input_list,y_train,epochs=10)

y_pred = model.predict(test_list)
# choose a optimal threshold
y_pred = y_pred>0.1