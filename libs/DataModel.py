
import pymongo
import pandas as pd




Mongo_ip = ''
Mongo_port = ''
Mongo_user = ''
Mongo_password = ''
Mongo_auth = ''

Mongo_uri = 'mongodb://' + Mongo_user + ':' + Mongo_password + '@' + Mongo_ip + ':' + Mongo_port + '/' + Mongo_auth

class DataMongo(object):

	def __init__(self):

		try:
			self.client = pymongo.MongoClient(Mongo_uri)

		except pymongo.errors.ConnectionFailure as e:
			print('Could not connect to server:', e)

	def get_ohlcv_data(self, dbName, collName):

		db = self.client[dbName][collName]
		cursor = db.find()
		df = pd.DataFrame(list(cursor)) 
		del df['_id']
		df = df.sort_values(['Time'], ascending = [True])
		
		return df

class DetailMongo(object):
	def __init__(self):

		try:
			self.client = pymongo.MongoClient(Mongo_uri)

		except pymongo.errors.ConnectionFailure as e:
			print('Could not connect to server:', e)

	def insertDetail(self, details, dbName, collName):
		db = self.client[dbName][collName]
		db.insert_many(details)

	def getStrike(self, dbName, collName):
		db = self.client[dbName][collName]
		cursor = db.find()
		if cursor.count() > 0:
			detail = cursor[0]
			db.remove(detail)
			detail.pop('_id')
			return detail

		else:
			return {}