import pandas as pd
import logging

logging.basicConfig(
    level=logging.INFO
)

FILENAME = "data/pp-2020.csv"

DATA_COLUMN = [
	"transactions_id","price","date_of_transfer","postcode","property_type",
	"old_new","duration","paon","saon","street","locality","town_city",
	"district","county","category","record_status"
]

LIMIT = 10000
CHUNKSIZE = 3000

PROJECT = "fashion-scraping"
DATASET = "ppd"
TABLE = "raw_events"

def get_csv_iterator(filename,chunksize,limit=None):
	if limit is None:
		return pd.read_csv(filename,chunksize=chunksize,header=0)
	else:
		return pd.read_csv(filename,chunksize=chunksize, header=0,nrows=limit)

def parse_post_code(post_str):
	try:
		post_list = post_str['postcode'].split(" ")
		post_list[1] = post_list[1][:1]
	except:
		print(post_str)
		raise Exception('parse_error')
	return post_list

def process_df(df,columns,project,dataset,table):
	df.columns = columns
	print(df.count())
	df.dropna(inplace=True,subset=['postcode'])
	df[['sector_1','sector_2']] = df[['postcode']].apply(parse_post_code,axis=1,result_type = 'expand')
	print(df.count())

"""
for i, df in enumerate(get_csv_iterator(FILENAME,CHUNKSIZE,LIMIT)):
	logging.info("Processing chunk {}".format(i+1))
	process_df(df,DATA_COLUMN,PROJECT,DATASET,TABLE)
"""

df = pd.read_csv(FILENAME,header=0)

process_df(df,DATA_COLUMN,PROJECT,DATASET,TABLE)

df.groupby(['sector_1','sector_2']).agg({"price":"sum"})

df.merge(how="left")