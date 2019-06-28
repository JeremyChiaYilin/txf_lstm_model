import requests
from bs4 import BeautifulSoup
from datetime import timedelta, date
import pymongo


import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib import rnn

import sys


sys.path.insert(0, '../libs/')

from lstm import LSTM
from DataModel import DataMongo
from DataMaker import DataMaker


def getTodayIndex(year, month, day, merchandise, market):
	# keys = {'syear': year, 'smonth': month, 'sday': day, 'market_code' : market} # 要傳給期貨交易所的 key
	keys = {'queryDate': '{}/{}/{}'.format(year, month, day), 'marketCode' : market, 'commodity_id' :merchandise , 'queryType':2}
	# r = requests.post("http://www.taifex.com.tw/chinese/3/3_1_1.asp", data = keys) # 抓取這個 key 回傳網頁的原始碼
	r = requests.post("http://www.taifex.com.tw/cht/3/futDailyMarketReport", data = keys)
	soup = BeautifulSoup(r.text, "lxml") # 把原始碼做整理
	# 這邊下面講
	
	soup_data = soup.select('table')[2].select('table')[1].select('td')
	# print(soup_data)
	return soup_data


def getTodayPosition(year, month, day):
	keys = {'syear': year, 'smonth': month, 'sday': day} # 要傳給期貨交易所的 key
	r = requests.post("http://www.taifex.com.tw/chinese/3/7_12_1.asp", data = keys) # 抓取這個 key 回傳網頁的原始碼
	soup = BeautifulSoup(r.text, "lxml") # 把原始碼做整理
	
	# res = soup.select('table')[2].select('table')[1].select('td')
	# for i in res:

	# 	print(int(i.text.replace(',', '')))

	# 這邊下面講
	soup_data = soup.select('table')[2].select('table')
	return soup_data

def getData(start, merchandise, market):

	# 先把 DataFrame 架構寫好
	data = {'Time':[], 'Open':[], 'High':[], 'Low':[], 'Close':[], 'Volume':[]}

	# 這邊匯入我們想要的資料的起始日期
	day = date(start[0], start[1], start[2])
	# 這邊把停止日期設置在今天
	today = date.today()
	# 起始日至今日的總日數
	span = ((today - day).days) + 1
	# 因為是每日資料，所以間隔設為一日
	interval = timedelta(days=1)

	# 多定義一個把爬下來的資料放進 DataFrame 的方法
	def addtoData(column, index):
		try:
			data[column].append(float(soup_data[index].text))
		except Exception as e:
			print (e)
			data[column].append(np.nan)

	# 把整個 span 內的 soup_data 中的當日期指各項資料抓下來
	for _ in range(span):
		try:
			soup_data = getTodayIndex(day.year, day.month, day.day, merchandise, market)
			data['Time'].append(day)
			addtoData('Open', 2)
			addtoData('High', 3)
			addtoData('Low', 4)
			addtoData('Close', 5)
			if market == 0:
				addtoData('Volume', 9)
			elif market == 1:
				addtoData('Volume', 8)
			
			print ('%d/%d/%d is loaded' %(day.year, day.month, day.day))
			
		except Exception as e:
			#print (e)
			print ('%s stock market was not open on %d/%d/%d' %(merchandise, day.year, day.month, day.day))
		# 日期往後加一天
		day += interval

	# 把 data 放到 DataFrame 裡面
	df = pd.DataFrame(data)
	return df

def getDataWithPosition(start, market):

	# 先把 DataFrame 架構寫好
	data = {'Time':[], 'Open':[], 'High':[], 'Low':[], 'Close':[], 'Volume':[],
			'sv':[], 'iv':[], 'fv':[], 'sp':[], 'ip':[], 'fp':[],}

	# 這邊匯入我們想要的資料的起始日期
	day = date(start[0], start[1], start[2])
	# 這邊把停止日期設置在今天
	today = date.today()
	# 起始日至今日的總日數
	span = ((today - day).days) + 1
	# 因為是每日資料，所以間隔設為一日
	interval = timedelta(days=1)

	# 多定義一個把爬下來的資料放進 DataFrame 的方法
	def addtoData(column, index):
		try:
			data[column].append(int(ohlcv_data[index].text))
		except Exception as e:
			print (e)
			data[column].append(np.nan)


	def addtoPosiData(column, table, index):
		try:
			data[column].append(int(position_data[table].select('td')[index].text.replace(',', '')))
		except Exception as e:
			print (e)
			data[column].append(np.nan)


	# 把整個 span 內的 soup_data 中的當日期指各項資料抓下來
	for _ in range(span):
		try:
			ohlcv_data = getTodayIndex(day.year, day.month, day.day, market)
			position_data = getTodayPosition(day.year, day.month, day.day)
			data['Time'].append(day)
			addtoData('Open', 2)
			addtoData('High', 3)
			addtoData('Low', 4)
			addtoData('Close', 5)

			if market == 0:
				addtoData('Volume', 9)
			elif market == 1:
				addtoData('Volume', 8)
			

			addtoPosiData('sv', 0, 4)
			addtoPosiData('iv', 0, 10)
			addtoPosiData('fv', 0, 16)
			addtoPosiData('sp', 1, 4)
			addtoPosiData('ip', 1, 10)
			addtoPosiData('fp', 1, 16)

			
			print ('%d/%d/%d is loaded' %(day.year, day.month, day.day))
			
		except Exception as e:
			#print (e)
			print ('stock market was not open on %d/%d/%d' %(day.year, day.month, day.day))
		# 日期往後加一天
		day += interval

	# 把 data 放到 DataFrame 裡面
	df = pd.DataFrame(data)
	return df



########################################################



Mongo_ip = ''
Mongo_port = ''
Mongo_user = ''
Mongo_password = ''
Mongo_auth = ''

Mongo_uri = 'mongodb://' + Mongo_user + ':' + Mongo_password + '@' + Mongo_ip + ':' + Mongo_port + '/' + Mongo_auth

client = pymongo.MongoClient(Mongo_uri)
db = client['DDPG_his_min']['TW_F_daily']

his = list(db.find())

his = pd.DataFrame(his)
del his['_id']
his = his.sort_values(by = 'Time', ascending = [True])
y = his.iloc[-1]['Time'].year
m = his.iloc[-1]['Time'].month
d = his.iloc[-1]['Time'].day + 1
#m = 6
#d = 1
print(his.iloc[-1]['Time'])
print(y, m, d)

# 從 1998/7/21 開始爬
# df = getData([1998,7,21], 0)
df = getData([y, m, d], 'TX', 0)
print(df)



df.Time = pd.to_datetime(df.Time)
print (df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']])

data_list = df.T.to_dict().values()
# print(data_list)

if len(data_list) != 0:
	db.insert_many(data_list)






MER = 'future'
order = '7-1'
CELL_SIZE = 8
STEPS = 20
LAB_STEPS = 5
threshold = 0.01
batchsize = 100
ma = ['5', '10', '20']

mongo = DataMongo()
df_ohlcv = mongo.get_ohlcv_data('DDPG_his_min', 'TW_F_daily')
df_ohlcv = df_ohlcv.set_index(['Time'])

maker = DataMaker()
df_inp = maker.parse_inp_data_ma(df_ohlcv, STEPS, ma)
df_lab = maker.parse_lab_data(df_ohlcv, LAB_STEPS, threshold)



modelName = 'tw_{}_lstm_daily_step{}_labstep{}_th{}_batch{}_repeat'.format(MER,
																		STEPS,
																		LAB_STEPS,
																		str(threshold).replace('.', ''),
																		batchsize)



_dir_best = '../model/lstm_ohlc_twf/FixedModel/{}_{}_best/'.format(modelName, order)
fileName_best = '{}_{}_best.ckpt'.format(modelName, 7)


model = LSTM(CELL_SIZE * STEPS, 3,
				CELL_SIZE, STEPS, False,
				isRestore = True,
				modelPath = _dir_best+fileName_best)

infos = []
count = 100
for i in range(count):
	date = df_inp.index[i-count]
	inp = [df_inp.iloc[i-count].values]
	op = df_ohlcv.Open.iloc[i-count]
	close = df_ohlcv.Close.iloc[i-count]

	signal, prob = model.predicting(inp)
	lab = df_lab.Label.loc[date]

	print('{}, prob : {}, signal : {}, label : {}, open : {}, close : {}'.format(date, prob, signal, lab, op, close))

