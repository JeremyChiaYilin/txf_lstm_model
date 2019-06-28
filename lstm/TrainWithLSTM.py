
import pandas as pd
import numpy as np
from sklearn.utils import shuffle

import tensorflow as tf
from tensorflow.contrib import rnn

import sys
from datetime import timedelta

sys.path.insert(0, '../libs/')

from AlgoBank import AlgoBank
from lstm import LSTM
from DataModel import DataMongo
from DataMaker import DataMaker
from BackTest import BackTest



class TrainModel(object):

	def __init__(self, CELL_SIZE, STEPS):

		self.CELL_SIZE = CELL_SIZE
		self.STEPS = STEPS

	def train(self, df_inp, df_lab, batches, epoch, test_num, modelPath = None, reuse = False):
		if modelPath is None:
			model = LSTM(self.CELL_SIZE * self.STEPS, 3, self.CELL_SIZE, self.STEPS, reuse)
		else:
			model = LSTM(self.CELL_SIZE * self.STEPS, 3,
						self.CELL_SIZE, self.STEPS, reuse,
						isRestore = True, modelPath = modelPath)
		


		df = pd.concat([df_inp, df_lab], axis = 1)
		
		df = df.dropna(how = 'any')
		
		sizes = len(df)
		
		delCols = ['ma_p', 'ma_n']

		for c in delCols:
			if c in df.columns:
				del df[c]

		df = shuffle(df)
		
		inp_datas = []
		lab_datas = []
		for i in range(sizes // batches):
			df_datas = df.iloc[i * batches : (i+1) * batches]

			inp = df_datas.drop(columns = ['Label']).values
			lab = df_datas['Label'].values

			inp_datas.append(inp)
			lab_datas.append(lab)

		test_inp = inp_datas[-1 * test_num : ]
		test_lab = lab_datas[-1 * test_num : ]

		del inp_datas[-1 * test_num : ]
		del lab_datas[-1 * test_num : ]

		for i in range(epoch):
			model.training(inp_datas, lab_datas)

			loss, accuracy, precision = model.testing(test_inp, test_lab)


		return model, loss, accuracy, precision


		## ensemble train method
	def train_2(self, df_data, df_test, batches, epoch, modelPath = None, reuse = False):
		if modelPath is None:
			model = LSTM(self.CELL_SIZE * self.STEPS, 3, self.CELL_SIZE, self.STEPS, reuse)
		else:
			model = LSTM(self.CELL_SIZE * self.STEPS, 3,
						self.CELL_SIZE, self.STEPS, reuse,
						isRestore = True, modelPath = modelPath)
		
		df_data_ = df_data.copy()
		sizes = len(df_data_)

		print('data size : ', sizes)

		inp_datas = []
		lab_datas = []
		for i in range(sizes // batches):
			df_datas = df_data_.iloc[i * batches : (i+1) * batches]

			inp = df_datas.drop(columns = ['Label']).values
			lab = df_datas['Label'].values

			inp_datas.append(inp)
			lab_datas.append(lab)


		df_test_ = df_test.copy()
		test_sizes = len(df_test_)

		test_inp = []
		test_lab = []

		inp = df_test_.drop(columns = ['Label']).values
		lab = df_test_['Label'].values

		test_inp.append(inp)
		test_lab.append(lab)

		
		for i in range(epoch):
			model.training(inp_datas, lab_datas)

			loss, accuracy, precision = model.testing(test_inp, test_lab)


		return model, loss, accuracy, precision


	

if __name__ == '__main__':

	MER = 'future'

	mongo = DataMongo()
	df_ohlcv = mongo.get_ohlcv_data('tw_pairtrading', 'TX')
	df_ohlcv = df_ohlcv.dropna(axis = 0, how = 'any').reset_index(drop = True)
	df_ohlcv = df_ohlcv.set_index(['Time'])

	start = '2000-01-01'
	end = '2018-02-28'
	mask = (df_ohlcv.index > start) & (df_ohlcv.index <= end)
	df_train = df_ohlcv.loc[mask]
	

	start = '2018-03-01'
	end = '2019-03-18'
	mask = (df_ohlcv.index > start) & (df_ohlcv.index <= end)
	df_backtest = df_ohlcv.loc[mask]

	


	##########################################################
	## calculate label counts

	# maker = DataMaker()

	# df_lab = maker.parse_lab_data_2(df_train, 5, 0.02).dropna(how = 'any')
	# res = maker.cal_lab_count(df_lab)
	# print(res)

	# df_lab = maker.extract_first_label(df_lab)
	# res = maker.cal_lab_count(df_lab)
	# print(res)



	# import time
	# time.sleep(600)

	##########################################################
	### restore model to backtest

	# maker = DataMaker()

	# order = 7
	# CELL_SIZE = 8
	# STEPS = 20
	# LAB_STEPS = 5
	# threshold = 0.01
	# batchsize = 100
	# ma = ['5', '10', '20']
		
	# df_inp = maker.parse_inp_data_ma(df_train, STEPS, ma)
	# df_lab = maker.parse_lab_data(df_train, LAB_STEPS, threshold)

	# df_inp_ = maker.parse_inp_data_ma(df_backtest, STEPS, ma)
	# df_lab_ = maker.parse_lab_data(df_backtest, LAB_STEPS, threshold)


	# # df_inp__ = maker.parse_inp_data_ma(df_backtest_, STEPS, ma)
	# # df_lab__ = maker.parse_lab_data(df_backtest_, LAB_STEPS, threshold)

	# modelName = 'tw_{}_lstm_daily_step{}_labstep{}_th{}_batch{}_repeat'.format(MER,
	# 																		STEPS,
	# 																		LAB_STEPS,
	# 																		str(threshold).replace('.', ''),
	# 																		batchsize)

	# # _dir = '/home/jeremy/model/lstm_ohlc_twf/validation/{}_{}/'.format(modelName, order)
	# # fileName = '{}_{}.ckpt'.format(modelName, order)

	# # model = LSTM(CELL_SIZE * STEPS, 3,
	# # 				CELL_SIZE, STEPS, False,
	# # 				isRestore = True,
	# # 				modelPath = _dir+fileName)


	# _dir_best = '/home/jeremy/model/lstm_ohlc_twf/FixedModel/{}_{}_best/'.format(modelName, order)
	# fileName_best = '{}_{}_best.ckpt'.format(modelName, order)

	
	# model = LSTM(CELL_SIZE * STEPS, 3,
	# 				CELL_SIZE, STEPS, False,
	# 				isRestore = True,
	# 				modelPath = _dir_best+fileName_best)

	
	# #loss = 0.004
	# #for loss in [0.001, 0.002, 0.003, 0.004, 0.005, 0.01, 0.02, 0.03]:
	# for loss in [0.01]:
	# 	for p in [0.01]:
	# 		backtest = BackTest(order, model, STEPS)
	# 		profit = backtest.backtest(df_backtest, df_inp_, df_lab_, loss, p)
	# 		print(loss, p, profit)

	# 		# backtest_ = BackTest(order, model, STEPS)
	# 		# profit_ = backtest_.backtest(df_backtest_, df_inp__, df_lab__, loss, p)
	# 		# print(loss, p, profit_)

	# backtest.save_history(modelName)
	# #backtest_.save_history(modelName + '_backtest')
	

	############################################################

	# ### repeatly train one model 

	maker = DataMaker()

	order = 10

	CELL_SIZE = 8
	STEPS = 20
	LAB_STEPS = 5
	threshold = 0.01
	batchsize = 100
	ma = ['5', '10', '20']
		
	df_inp = maker.parse_inp_data_ma(df_train, STEPS, ma)
	df_lab = maker.parse_lab_data(df_train, LAB_STEPS, threshold)
	df_lab = maker.extract_first_label(df_lab)

	df_inp_ = maker.parse_inp_data_ma(df_backtest, STEPS, ma)
	df_lab_ = maker.parse_lab_data(df_backtest, LAB_STEPS, threshold)
	df_lab_ = maker.extract_first_label(df_lab_)


	modelName = 'tw_{}_lstm_daily_step{}_labstep{}_th{}_batch{}_repeat'.format(MER,
																			STEPS,
																			LAB_STEPS,
																			str(threshold).replace('.', ''),
																			batchsize)
	_dir = '../model/lstm_ohlc_twf/validation/{}_{}/'.format(modelName, order)
	fileName = '{}_{}.ckpt'.format(modelName, order)

	_dir_best = '../model/lstm_ohlc_twf/validation/{}_{}_best/'.format(modelName, order)
	fileName_best = '{}_{}_best.ckpt'.format(modelName, order)

	import atexit
	global infos
	infos = []
	def exit():
		global infos
		if infos:
			infos = pd.DataFrame(infos)
			cols = ['name', 'loss', 'accuracy', 'precision', 'validation', 'backtest']
			infos = infos.reindex(columns = cols)
			infos.to_csv('./info/{}_{}.csv'.format(modelName, order))
		print('infos : ', infos)
	atexit.register(exit)


	df_data = maker.gen_ensemble_data(df_inp, df_lab)
	df_test = maker.gen_ensemble_data(df_inp_, df_lab_)


	trainer = TrainModel(CELL_SIZE, STEPS)
	model, loss, accuracy, precision = trainer.train_2(df_data, df_test, batchsize, 1, reuse = False) ## batches, epoch, test_num

	backtest = BackTest(order, model, STEPS)
	validation_profit = backtest.backtest(df_backtest, df_inp_, df_lab_, 0.004, 0.03)
	best_loss = loss

	print('validaton : ', validation_profit)

	

	model.save_model(_dir, fileName)
	model.close()

	


	for i in range(600):

		trainer = TrainModel(CELL_SIZE, STEPS)
		model, loss, accuracy, precision = trainer.train_2(df_data, df_test, 100, 1, modelPath = _dir + fileName, reuse = False) ## batches, epoch, test_num

		backtest = BackTest(order, model, STEPS)
		profit = backtest.backtest(df_backtest, df_inp_, df_lab_, 0.004, 0.03)
		print('validaton : ', profit)

	

		if True:
			model.save_model(_dir, fileName)
			# if profit > validation_profit:
			if loss < best_loss:
				model.save_model(_dir_best, fileName_best)
				print('#### get changed ####')
				validation_profit = profit
				best_loss = loss
				info = {}
				info['name'] = str(i)
				info['loss'] = loss
				info['accuracy'] = accuracy
				info['precision'] = precision
				info['validation'] = profit
				infos.append(info)
				print('#### Best Profit {} ####'.format(validation_profit))


		model.close()
		
		print('{}_{} - {}'.format(modelName, order, i))



	