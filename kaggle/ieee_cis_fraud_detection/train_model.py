import pandas as pd
import numpy as np

import os
import pprint

import tensorflow as tf
import apache_beam as beam
import tensorflow_transform as tft
import tensorflow_transform.beam as tft_beam

train = 'data/adult.data'
test = 'data/adult.test'

CATEGORICAL_FEATURE_KEYS = [
    'workclass',
    'education',
    'marital-status',
    'occupation',
    'relationship',
    'race',
    'sex',
    'native-country',
]
NUMERIC_FEATURE_KEYS = [
    'age',
    'capital-gain',
    'capital-loss',
    'hours-per-week',
]
OPTIONAL_NUMERIC_FEATURE_KEYS = [
    'education-num',
]
LABEL_KEY = 'label'

RAW_DATA_FEATURE_SPEC = dict(
    [(name, tf.io.FixedLenFeature([], tf.string))
     for name in CATEGORICAL_FEATURE_KEYS] +
    [(name, tf.io.FixedLenFeature([], tf.float32))
     for name in NUMERIC_FEATURE_KEYS] +
    [(name, tf.io.VarLenFeature(tf.float32))
     for name in OPTIONAL_NUMERIC_FEATURE_KEYS] +
    [(LABEL_KEY, tf.io.FixedLenFeature([], tf.string))]
)

RAW_DATA_METADATA = tft.tf_metadata.dataset_metadata.DatasetMetadata(
    tft.tf_metadata.dataset_schema.schema_utils.schema_from_feature_spec(RAW_DATA_FEATURE_SPEC))

TRAIN_NUM_EPOCHS = 16
NUM_TRAIN_INSTANCES = 32561
TRAIN_BATCH_SIZE = 128
NUM_TEST_INSTANCES = 16281

# Names of temp files
TRANSFORMED_TRAIN_DATA_FILEBASE = 'train_transformed'
TRANSFORMED_TEST_DATA_FILEBASE = 'test_transformed'
EXPORTED_MODEL_DIR = 'exported_model_dir'

class MapAndFilterErrors(beam.PTransform):
  """Like beam.Map but filters out errors in the map_fn."""

  class _MapAndFilterErrorsDoFn(beam.DoFn):
    """Count the bad examples using a beam metric."""

    def __init__(self, fn):
      self._fn = fn
      # Create a counter to measure number of bad elements.
      self._bad_elements_counter = beam.metrics.Metrics.counter(
          'census_example', 'bad_elements')

    def process(self, element):
      try:
        yield self._fn(element)
      except Exception:  # pylint: disable=broad-except
        # Catch any exception the above call.
        self._bad_elements_counter.inc(1)

  def __init__(self, fn):
    self._fn = fn

  def expand(self, pcoll):
    return pcoll | beam.ParDo(self._MapAndFilterErrorsDoFn(self._fn))

def preprocessing_fn(inputs):
  """Preprocess input columns into transformed columns."""
  # Since we are modifying some features and leaving others unchanged, we
  # start by setting `outputs` to a copy of `inputs.
  outputs = inputs.copy()

  # Scale numeric columns to have range [0, 1].
  for key in NUMERIC_FEATURE_KEYS:
    outputs[key] = tft.scale_to_0_1(outputs[key])

  for key in OPTIONAL_NUMERIC_FEATURE_KEYS:
    # This is a SparseTensor because it is optional. Here we fill in a default
    # value when it is missing.
    sparse = tf.sparse.SparseTensor(outputs[key].indices, outputs[key].values,
                                    [outputs[key].dense_shape[0], 1])
    dense = tf.sparse.to_dense(sp_input=sparse, default_value=0.)
    # Reshaping from a batch of vectors of size 1 to a batch to scalars.
    dense = tf.squeeze(dense, axis=1)
    outputs[key] = tft.scale_to_0_1(dense)

  # For all categorical columns except the label column, we generate a
  # vocabulary but do not modify the feature.  This vocabulary is instead
  # used in the trainer, by means of a feature column, to convert the feature
  # from a string to an integer id.
  for key in CATEGORICAL_FEATURE_KEYS:
    tft.vocabulary(inputs[key], vocab_filename=key)

  # For the label column we provide the mapping from string to index.
  table_keys = ['>50K', '<=50K']
  initializer = tf.lookup.KeyValueTensorInitializer(
      keys=table_keys,
      values=tf.cast(tf.range(len(table_keys)), tf.int64),
      key_dtype=tf.string,
      value_dtype=tf.int64)
  table = tf.lookup.StaticHashTable(initializer, default_value=-1)
  outputs[LABEL_KEY] = table.lookup(outputs[LABEL_KEY])

  return outputs

def transform_data(train_data_file, test_data_file, working_dir):
  """Transform the data and write out as a TFRecord of Example protos.

  Read in the data using the CSV reader, and transform it using a
  preprocessing pipeline that scales numeric data and converts categorical data
  from strings to int64 values indices, by creating a vocabulary for each
  category.

  Args:
    train_data_file: File containing training data
    test_data_file: File containing test data
    working_dir: Directory to write transformed data and metadata to
  """

  # The "with" block will create a pipeline, and run that pipeline at the exit
  # of the block.
  with beam.Pipeline() as pipeline:
    with tft_beam.Context(temp_dir=tempfile.mkdtemp()):
      # Create a coder to read the census data with the schema.  To do this we
      # need to list all columns in order since the schema doesn't specify the
      # order of columns in the csv.
      ordered_columns = [
          'age', 'workclass', 'fnlwgt', 'education', 'education-num',
          'marital-status', 'occupation', 'relationship', 'race', 'sex',
          'capital-gain', 'capital-loss', 'hours-per-week', 'native-country',
          'label'
      ]
      converter = tft.coders.CsvCoder(ordered_columns, RAW_DATA_METADATA.schema)

      # Read in raw data and convert using CSV converter.  Note that we apply
      # some Beam transformations here, which will not be encoded in the TF
      # graph since we don't do them from within tf.Transform's methods
      # (AnalyzeDataset, TransformDataset etc.).  These transformations are just
      # to get data into a format that the CSV converter can read, in particular
      # removing spaces after commas.
      #
      # We use MapAndFilterErrors instead of Map to filter out decode errors in
      # convert.decode which should only occur for the trailing blank line.
      raw_data = (
          pipeline
          | 'ReadTrainData' >> beam.io.ReadFromText(train_data_file)
          | 'FixCommasTrainData' >> beam.Map(
              lambda line: line.replace(', ', ','))
          | 'DecodeTrainData' >> MapAndFilterErrors(converter.decode))

      # Combine data and schema into a dataset tuple.  Note that we already used
      # the schema to read the CSV data, but we also need it to interpret
      # raw_data.
      raw_dataset = (raw_data, RAW_DATA_METADATA)
      transformed_dataset, transform_fn = (
          raw_dataset | tft_beam.AnalyzeAndTransformDataset(preprocessing_fn))
      transformed_data, transformed_metadata = transformed_dataset
      transformed_data_coder = tft.coders.ExampleProtoCoder(
          transformed_metadata.schema)

      _ = (
          transformed_data
          | 'EncodeTrainData' >> beam.Map(transformed_data_coder.encode)
          | 'WriteTrainData' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE)))

      # Now apply transform function to test data.  In this case we remove the
      # trailing period at the end of each line, and also ignore the header line
      # that is present in the test data file.
      raw_test_data = (
          pipeline
          | 'ReadTestData' >> beam.io.ReadFromText(test_data_file,
                                                   skip_header_lines=1)
          | 'FixCommasTestData' >> beam.Map(
              lambda line: line.replace(', ', ','))
          | 'RemoveTrailingPeriodsTestData' >> beam.Map(lambda line: line[:-1])
          | 'DecodeTestData' >> MapAndFilterErrors(converter.decode))

      raw_test_dataset = (raw_test_data, RAW_DATA_METADATA)

      transformed_test_dataset = (
          (raw_test_dataset, transform_fn) | tft_beam.TransformDataset())
      # Don't need transformed data schema, it's the same as before.
      transformed_test_data, _ = transformed_test_dataset

      _ = (
          transformed_test_data
          | 'EncodeTestData' >> beam.Map(transformed_data_coder.encode)
          | 'WriteTestData' >> beam.io.WriteToTFRecord(
              os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE)))

      # Will write a SavedModel and metadata to working_dir, which can then
      # be read by the tft.TFTransformOutput class.
      _ = (
          transform_fn
          | 'WriteTransformFn' >> tft_beam.WriteTransformFn(working_dir))

def _make_training_input_fn(tf_transform_output, transformed_examples,
                            batch_size):
  """Creates an input function reading from transformed data.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.
    transformed_examples: Base filename of examples.
    batch_size: Batch size.

  Returns:
    The input function for training or eval.
  """
  def input_fn():
    """Input function for training and eval."""
    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=transformed_examples,
        batch_size=batch_size,
        features=tf_transform_output.transformed_feature_spec(),
        reader=tf.data.TFRecordDataset,
        shuffle=True)

    transformed_features = tf.compat.v1.data.make_one_shot_iterator(
        dataset).get_next()

    # Extract features and label from the transformed tensors.
    transformed_labels = transformed_features.pop(LABEL_KEY)

    return transformed_features, transformed_labels

  return input_fn

def _make_serving_input_fn(tf_transform_output):
  """Creates an input function reading from raw data.

  Args:
    tf_transform_output: Wrapper around output of tf.Transform.

  Returns:
    The serving input function.
  """
  raw_feature_spec = RAW_DATA_FEATURE_SPEC.copy()
  # Remove label since it is not available during serving.
  raw_feature_spec.pop(LABEL_KEY)

  def serving_input_fn():
    """Input function for serving."""
    # Get raw features by generating the basic serving input_fn and calling it.
    # Here we generate an input_fn that expects a parsed Example proto to be fed
    # to the model at serving time.  See also
    # tf.estimator.export.build_raw_serving_input_receiver_fn.
    raw_input_fn = tf.estimator.export.build_parsing_serving_input_receiver_fn(
        raw_feature_spec, default_batch_size=None)
    serving_input_receiver = raw_input_fn()

    # Apply the transform function that was used to generate the materialized
    # data.
    raw_features = serving_input_receiver.features
    transformed_features = tf_transform_output.transform_raw_features(
        raw_features)

    return tf.estimator.export.ServingInputReceiver(
        transformed_features, serving_input_receiver.receiver_tensors)

  return serving_input_fn

def get_feature_columns(tf_transform_output):
  """Returns the FeatureColumns for the model.

  Args:
    tf_transform_output: A `TFTransformOutput` object.

  Returns:
    A list of FeatureColumns.
  """
  # Wrap scalars as real valued columns.
  real_valued_columns = [tf.feature_column.numeric_column(key, shape=())
                         for key in NUMERIC_FEATURE_KEYS]

  # Wrap categorical columns.
  one_hot_columns = [
      tf.feature_column.categorical_column_with_vocabulary_file(
          key=key,
          vocabulary_file=tf_transform_output.vocabulary_file_by_name(
              vocab_filename=key))
      for key in CATEGORICAL_FEATURE_KEYS]

  return real_valued_columns + one_hot_columns
def train_and_evaluate(working_dir, num_train_instances=NUM_TRAIN_INSTANCES,
                       num_test_instances=NUM_TEST_INSTANCES):
  """Train the model on training data and evaluate on test data.

  Args:
    working_dir: Directory to read transformed data and metadata from and to
        write exported model to.
    num_train_instances: Number of instances in train set
    num_test_instances: Number of instances in test set

  Returns:
    The results from the estimator's 'evaluate' method
  """
  tf_transform_output = tft.TFTransformOutput(working_dir)

  run_config = tf.estimator.RunConfig()

  estimator = tf.estimator.LinearClassifier(
      feature_columns=get_feature_columns(tf_transform_output),
      config=run_config,
      loss_reduction=tf.losses.Reduction.SUM)

  # Fit the model using the default optimizer.
  train_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, TRANSFORMED_TRAIN_DATA_FILEBASE + '*'),
      batch_size=TRAIN_BATCH_SIZE)
  estimator.train(
      input_fn=train_input_fn,
      max_steps=TRAIN_NUM_EPOCHS * num_train_instances / TRAIN_BATCH_SIZE)

  # Evaluate model on test dataset.
  eval_input_fn = _make_training_input_fn(
      tf_transform_output,
      os.path.join(working_dir, TRANSFORMED_TEST_DATA_FILEBASE + '*'),
      batch_size=1)

  # Export the model.
  serving_input_fn = _make_serving_input_fn(tf_transform_output)
  exported_model_dir = os.path.join(working_dir, EXPORTED_MODEL_DIR)
  estimator.export_saved_model(exported_model_dir, serving_input_fn)

  return estimator.evaluate(input_fn=eval_input_fn, steps=num_test_instances)


import tempfile
temp = tempfile.gettempdir()

transform_data(train, test, temp)
results = train_and_evaluate(temp)
pprint.pprint(results)
'''

#from imblearn.over_sampling import SMOTE


FILES = {
	"test": "data/act_test.csv"
}

train = pd.read_csv("data/train_transaction.csv")
#test = pd.read_csv("data/test_transaction.csv")


train.columns

test[['TransactionID', 'TransactionDT', 'TransactionAmt', 'ProductCD', 
'card1', 'card2', 'card3', 'card4', 'card5', 'card6', 'addr1', 'addr2', 'dist1', 'dist2', 
'P_emaildomain', 'R_emaildomain', 'C1', 'C10','D1', 'D2','D10', 'D11', 'M1', 'M2', 'M8', 
'M9', 'V1', 'V2', 'V12', 'V13', ]].sample(15).values.tolist()
print(train.sample(15))

# generate time of day


train["Time of Day"] = np.floor(train["TransactionDT"]/3600/183)
test["Time of Day"] = np.floor(test["TransactionDT"]/3600/183)

# drop columns
train.drop("TransactionDT",axis=1,inplace=True)
test.drop("TransactionDT",axis=1,inplace=True)

# define continuous and categorical variables
cont_vars = ["TransactionAmt"]
cat_vars = ["ProductCD","addr1","addr2","P_emaildomain","R_emaildomain","Time of Day"] + [col for col in train.columns if "card" in col]

# set training and testing set
x_train = train[cont_vars + cat_vars].copy()
y_train = train["isFraud"].copy()
x_test = train[cont_vars + cat_vars].copy()
y_test = train["isFraud"].copy()

# process cont_vars
# scale values

scaler = StandardScaler()
x_train["TransactionAmt"] = scaler.fit_transform(x_train["TransactionAmt"].values.reshape(-1,1))
x_test["TransactionAmt"] = scaler.transform(x_test["TransactionAmt"].values.reshape(-1,1))

# reduce cardinality of categorical variables
idx_list = x_train["card1"].value_counts()[x_train["card1"].value_counts()<=100].index.tolist()
x_train.loc[x_train["card1"].isin(idx_list),"card1"] = "Others"
x_test.loc[x_test["card1"].isin(idx_list),"card1"] = "Others"

# fill missing
x_train[cat_vars] = x_train[cat_vars].fillna("Missing")
x_test[cat_vars] = x_test[cat_vars].fillna("Missing")

def categorify(df, cat_vars):
	categories = {}
	for cat in cat_vars:
		df[cat] = df[cat].astype("category").cat.as_ordered()
		categories[cat] = df[cat].cat.categories
	return categories

def apply_test(test,categories):
	for cat, index in categories.items():
		test[cat] = pd.Categorical(test[cat],categories=categories[cat],ordered=True)

categories = categorify(x_train,cat_vars)

apply_test(x_test,categories)

for cat in cat_vars:
	x_train[cat] = x_train[cat].cat.codes+1
	x_test[cat] = x_test[cat].cat.codes+1


sm = SMOTE(random_state=0)
x_sm, y_train = sm.fit_resample(x_train, y_train)
x_train = pd.DataFrame(x_sm,columns=x_train.columns)
print(x_train.head())
'''