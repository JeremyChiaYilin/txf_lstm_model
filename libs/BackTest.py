
import pandas as pd
import numpy as np
from DataModel import DetailMongo

OpenPrice = 'Open_price'
ClosePrice = 'Close_price'
OpenTime = 'Open_time'
CloseTime = 'Close_time'
Volume = 'Volume'
Trend = 'Trend'
Profit = 'Profit'

Contract = 200
#commision = self.strike[OpenPrice] * Contract * 0.000012
commision = 100 * 2

class BackTest(object):

	def __init__(self, order, model, STEPS):

		self.order = order
		self.model = model
		self.STEPS = STEPS
		
		self.hasPosition = False
		self.strike = {}
		self.history = []

		self.preSignal = 0
		self.isLossFusing = False

		self.flatCount = 0

		self.dm = DetailMongo()


	def Signal(self, inp):

		signal = self.model.predicting(inp)[0]

		return signal

	def QtySignal(self, ohlcv):

		if ohlcv['Close'] > ohlcv['Open'] and ohlcv['Close'] > ohlcv['10']:
			return 1
		elif ohlcv['Close'] < ohlcv['Open'] and ohlcv['Close'] < ohlcv['10']:
			return 2
		else:
			return 0


	def backtest(self, df_ohlcv, df_inp, df_lab, loss, profit):

		drop = df_inp[df_inp.columns[-1]].isnull().sum()
		df_inp = df_inp.reset_index()[drop : ]
		df_ohlcv = df_ohlcv.reset_index()[drop : ]
		df_lab = df_lab.reset_index()[drop : ]

		sizes = len(df_inp)

		del df_inp['Time']

		for i in range(sizes-1):

			inp = [df_inp.iloc[i].values]
			ohlcv = df_ohlcv.iloc[i+1]
			signal = self.Signal(inp)

			
			lab = df_lab.iloc[i]['Label']
			#print(lab, signal, ohlcv['Close'])
			
			#self.Trade(signal, ohlcv, loss, profit)
			self.Trade(signal, ohlcv, loss)


			# ohlcv_ = df_ohlcv.iloc[i]
			# qtySignal = self.QtySignal(ohlcv_)
			# self.Trade(signal, qtySignal, ohlcv, loss)

		profit = 0
		for h in self.history:
			profit += h[Profit]
		
		return profit


	def save_history(self, modelName):
		if len(self.history) > 0:	
			self.dm.insertDetail(self.history, 'LSTM_detail', '{}_{}'.format(modelName, self.order))
		else:
			print('No history details to save')

	## qty signal no stop profit
	# def Trade(self, signal, qtySignal, ohlcv, loss):

	# 	if self.preSignal != signal:
	# 		self.isLossFusing = False

	# 	self.preSignal = signal

	# 	if not self.hasPosition:
	# 		if not self.isLossFusing:
	# 			if signal == 1 and qtySignal == 1:
	# 				self.OpenPosition(ohlcv, 'buy')
	# 			elif signal == 2 and qtySignal == 2:
	# 				self.OpenPosition(ohlcv, 'sell')

	# 	else:
	# 		if self.isStopLoss(ohlcv, loss):
	# 			self.ClosePosition(ohlcv)
	# 			self.isLossFusing = True
	# 		else:
	# 			if signal == 1 and qtySignal == 1 and self.strike[Trend] == 'sell':
	# 				self.ClosePosition(ohlcv)
	# 			elif signal == 2 and qtySignal == 2 and self.strike[Trend] == 'buy':
	#				self.ClosePosition(ohlcv)

	# def Trade(self, signal, ohlcv, loss, profit):

	# 	if self.preSignal != signal:
	# 		self.isLossFusing = False

	# 	self.preSignal = signal

	# 	if not self.hasPosition:
	# 		if not self.isLossFusing:
	# 			if signal == 1:
	# 				self.OpenPosition(ohlcv, 'buy')
	# 			elif signal == 2:
	# 				self.OpenPosition(ohlcv, 'sell')

	# 	else:
	# 		if self.isStopLoss(ohlcv, loss):
	# 			self.ClosePosition(ohlcv)
	# 			self.isLossFusing = True
	# 		elif self.isStopProfit(ohlcv, profit):
	# 			self.ClosePosition(ohlcv)
	# 			#self.isLossFusing = True
	# 		else:
	# 			if signal == 1 and self.strike[Trend] == 'sell':
	# 				self.ClosePosition(ohlcv)
	# 			elif signal == 2 and self.strike[Trend] == 'buy':
	# 				self.ClosePosition(ohlcv)

	## no stop profit
	def Trade(self, signal, ohlcv, loss):

		if self.preSignal != signal:
			self.isLossFusing = False

		self.preSignal = signal

		if not self.hasPosition:
			if not self.isLossFusing:
				if signal == 1:
					self.OpenPosition(ohlcv, 'buy')
				elif signal == 2:
					self.OpenPosition(ohlcv, 'sell')

		else:
			if self.isStopLoss(ohlcv, loss):
				self.ClosePosition(ohlcv)
				self.isLossFusing = True
			else:
				if signal == 1 and self.strike[Trend] == 'sell':
					self.ClosePosition(ohlcv)
				elif signal == 2 and self.strike[Trend] == 'buy':
					self.ClosePosition(ohlcv)
				
				# elif signal == 0:
				# 	if self.flatCount < 5:
				# 		self.flatCount += 1
				# 	else:
				# 		self.ClosePosition(ohlcv)
				# 		self.flatCount = 0

	## no fuse
	# def Trade(self, signal, ohlcv, loss):

	# 	if not self.hasPosition:
	# 		if signal == 1:
	# 			self.OpenPosition(ohlcv, 'buy')
	# 		elif signal == 2:
	# 			self.OpenPosition(ohlcv, 'sell')

	# 	else:
	# 		if self.isStopLoss(ohlcv, loss):
	# 			self.ClosePosition(ohlcv)
	# 		else:
	# 			if signal == 1 and self.strike[Trend] == 'sell':
	# 				self.ClosePosition(ohlcv)
	# 			elif signal == 2 and self.strike[Trend] == 'buy':
	# 				self.ClosePosition(ohlcv)

	## no stop loss 
	# def Trade(self, signal, ohlcv):

	# 	if not self.hasPosition:
	# 		if signal == 1:
	# 			self.OpenPosition(ohlcv, 'buy')
	# 		elif signal == 2:
	# 			self.OpenPosition(ohlcv, 'sell')

	# 	else:
	# 		if signal == 1 and self.strike[Trend] == 'sell':
	# 			self.ClosePosition(ohlcv)
	# 			self.OpenPosition(ohlcv, 'buy')
	# 		elif signal == 2 and self.strike[Trend] == 'buy':
	# 			self.ClosePosition(ohlcv)
	# 			self.OpenPosition(ohlcv, 'sell')

	def isStopProfit(self, ohlcv, profit):
		price = ohlcv['Open']
		if self.strike[Trend] == 'buy':
			if price > self.strike[OpenPrice] * (1 + profit):
				return True
			else:
				return False
		elif self.strike[Trend] == 'sell':
			if price < self.strike[OpenPrice] * (1 - profit):
				return True
			else:
				return False
		return False

	def isStopLoss(self, ohlcv, loss):
		price = ohlcv['Open']
		if self.strike[Trend] == 'buy':
			#if price < self.strike[OpenPrice] * (1 - 0.005):
			if price < self.strike[OpenPrice] * (1 - loss):
				return True
			else:
				return False
		elif self.strike[Trend] == 'sell':
			#if price > self.strike[OpenPrice] * (1 + 0.005):
			if price > self.strike[OpenPrice] * (1 + loss):
				return True
			else:
				return False
		return False

	def OpenPosition(self, ohlcv, b_s):

		self.strike[OpenTime] = ohlcv['Time']
		self.strike[OpenPrice] = int(ohlcv['Open'])
		self.strike[Volume] = 1
		self.strike[Trend] = b_s
		self.hasPosition = True

	def ClosePosition(self, ohlcv):

		self.strike[CloseTime] = ohlcv['Time']
		self.strike[ClosePrice] = int(ohlcv['Open'])
		
		if self.strike[Trend] == 'buy':
			self.strike[Profit] = (self.strike[ClosePrice] - self.strike[OpenPrice]) * Contract - commision
		elif self.strike[Trend] == 'sell':
			self.strike[Profit] = (self.strike[OpenPrice] - self.strike[ClosePrice]) * Contract - commision

		self.history.append(self.strike)

		self.strike = {}
		self.hasPosition = False

