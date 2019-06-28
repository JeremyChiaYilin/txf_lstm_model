
import pandas as pd
from sklearn.utils import shuffle


class DataMaker(object):


	def normalize(self, data):

		mean = data.mean(axis = 1)
		std = data.std(axis = 1)
		
		for column in data:

			data[column] = (data[column] - mean) / std

		return data


	def cal_cost_price(self, df_ohlcv, period):

		df = df_ohlcv.copy()

		df['Price'] = (df['Open'] + df['Close']) / 2

		for i in range(period):
			if i == 0:
				df['TotalVolume'] = df['Volume'].shift(i)
				df['TotalPrice'] = df['Price'].shift(i) * df['Volume'].shift(i)
			else:
				df['TotalVolume'] += df['Volume'].shift(i)
				df['TotalPrice'] += df['Price'].shift(i) * df['Volume'].shift(i)

		df['Cost'] = df['TotalPrice'] / df['TotalVolume']
		
		return df['Cost']


	def cal_lab_count(self, df):
	
		counts = df['Label'].value_counts()
		
		return counts


	def parse_inp_data(self, df_ohlcv, steps):
	
		df_ohlc = df_ohlcv[['Open', 'High', 'Low', 'Close']].copy()
		df_v = df_ohlcv['Volume'].to_frame().copy()
	
		col_ohlc = df_ohlc.columns
		col_v = df_v.columns

		cols_ = ['Open', 'High', 'Low', 'Close', 'Volume']
		cols = ['Open', 'High', 'Low', 'Close', 'Volume']
		for i in range(1, steps):
			for col in col_ohlc:
				df_ohlc[ col + '_' + str(i)] = df_ohlc[col].shift(i)
			
			for col in col_v:
				df_v[ col + '_' + str(i)] = df_v[col].shift(i)
			
			for col in cols_:
				cols.append(col + '_' + str(i))

		df_ohlc = self.normalize(df_ohlc)
		df_v = self.normalize(df_v)
		
		df = df.reindex(columns = cols)

		return df

	def parse_inp_data_cost(self, df_ohlcv, steps, ma):

		for i in ma:
			df_ohlcv[i] = self.cal_cost_price(df_ohlcv, int(i))

		df_p = df_ohlcv[['Open', 'High', 'Low', 'Close'] + ma].copy()
		df_v = df_ohlcv['Volume'].to_frame().copy()
	
		col_p = df_p.columns
		col_v = df_v.columns

		cols_ = ['Open', 'High', 'Low', 'Close', 'Volume'] + ma
		cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + ma
		for i in range(1, steps):
			for col in col_p:
				df_p[ col + '_' + str(i)] = df_p[col].shift(i)
			
			for col in col_v:
				df_v[ col + '_' + str(i)] = df_v[col].shift(i)
			
			for col in cols_:
				cols.append(col + '_' + str(i))

		df_p = self.normalize(df_p)
		df_v = self.normalize(df_v)
		df = pd.concat([df_p, df_v], axis = 1)
		
		df = df.reindex(columns = cols)

		return df

	def parse_inp_data_ma(self, df_ohlcv, steps, ma):

		for i in ma:
			df_ohlcv[i] = df_ohlcv['Close'].rolling(center = False,
													window = int(i)).mean()

		df_p = df_ohlcv[['Open', 'High', 'Low', 'Close'] + ma].copy()
		df_v = df_ohlcv['Volume'].to_frame().copy()
	
		col_p = df_p.columns
		col_v = df_v.columns

		cols_ = ['Open', 'High', 'Low', 'Close', 'Volume'] + ma
		cols = ['Open', 'High', 'Low', 'Close', 'Volume'] + ma
		for i in range(1, steps):
			for col in col_p:
				df_p[ col + '_' + str(i)] = df_p[col].shift(i)
			
			for col in col_v:
				df_v[ col + '_' + str(i)] = df_v[col].shift(i)
			
			for col in cols_:
				cols.append(col + '_' + str(i))

		df_p = self.normalize(df_p)
		df_v = self.normalize(df_v)
		df = pd.concat([df_p, df_v], axis = 1)
		
		df = df.reindex(columns = cols)

		return df

	def parse_inp_data_ma_p(self, df_ohlcv, steps, ma):

		for i in ma:
			df_ohlcv[i] = df_ohlcv['Close'].rolling(center = False,
													window = int(i)).mean()

		df_p = df_ohlcv[['Open', 'High', 'Low', 'Close'] + ma].copy()
		df_v = df_ohlcv['Volume'].to_frame().copy()
		df_pv = df_ohlcv[['fv', 'iv', 'sv', 'fp', 'ip', 'sp']].copy()

		col_p = df_p.columns
		col_v = df_v.columns
		col_pv = df_pv.columns

		cols_ = ['Open', 'High', 'Low', 'Close', 'Volume', 'fv', 'iv', 'sv', 'fp', 'ip', 'sp'] + ma
		cols = ['Open', 'High', 'Low', 'Close', 'Volume', 'fv', 'iv', 'sv', 'fp', 'ip', 'sp'] + ma
		for i in range(1, steps):
			for col in col_p:
				df_p[ col + '_' + str(i)] = df_p[col].shift(i)
			
			for col in col_v:
				df_v[ col + '_' + str(i)] = df_v[col].shift(i)
			
			for col in col_pv:
				df_pv[ col + '_' + str(i)] = df_pv[col].shift(i)

			for col in cols_:
				cols.append(col + '_' + str(i))

		df_p = self.normalize(df_p)
		df_v = self.normalize(df_v)
		df_pv = self.normalize(df_pv)
		df = pd.concat([df_p, df_v, df_pv], axis = 1)
		
		df = df.reindex(columns = cols)

		return df



	def parse_inp_data_ma_nov(self, df_ohlcv, steps, ma):

		for i in ma:
			df_ohlcv[i] = df_ohlcv['Close'].rolling(center = False,
													window = int(i)).mean()

		df_p = df_ohlcv[['Open', 'High', 'Low', 'Close'] + ma].copy()
		
		col_p = df_p.columns
	
		cols_ = ['Open', 'High', 'Low', 'Close'] + ma
		cols = ['Open', 'High', 'Low', 'Close'] + ma
		for i in range(1, steps):
			for col in col_p:
				df_p[ col + '_' + str(i)] = df_p[col].shift(i)
			
			
			for col in cols_:
				cols.append(col + '_' + str(i))

		df = self.normalize(df_p)
	
		df = df.reindex(columns = cols)

		return df

	def parse_lab_data(self, df_ohlcv, steps, threshold):

		df_ohlcv = df_ohlcv.reset_index()
	
		df_c = df_ohlcv[['Time', 'Close']].reset_index(drop = True)
		df_c_ = df_c.sort_index(ascending = False).reset_index(drop = True)

		df_c['ma_p'] = df_c.Close.rolling(center = False, window = steps).mean()

		df_c_['ma_n'] = df_c_.Close.rolling(center = False, window = steps).mean()
		df_c_ = df_c_.sort_index(ascending = False).reset_index(drop = True)


		df_c = df_c.set_index(['Time'])
		df_c_ = df_c_.set_index(['Time'])

		df = pd.concat([df_c.ma_p, df_c_.ma_n], axis = 1)

		df['Label'] = df.apply(lambda x : 1 if x['ma_n'] > x['ma_p']*(1+threshold) else (2 if x['ma_n'] < x['ma_p']*(1-threshold) else 0), axis = 1)
		
		#df = df['Label'].to_frame()

		return df


	def parse_lab_data_2(self, df_ohlcv, steps, threshold):

		df_ohlcv = df_ohlcv.reset_index()

		df_c = df_ohlcv[['Time', 'Close']].reset_index(drop = True)
		df_c_ = df_c.sort_index(ascending = False).reset_index(drop = True)

		df_c_['ma_n'] = df_c_.Close.rolling(center = False, window = steps).mean()
		df_c_ = df_c_.sort_index(ascending = False).reset_index(drop = True)

		df_c = df_c.set_index(['Time'])
		df_c_ = df_c_.set_index(['Time'])

		df = pd.concat([df_c.Close, df_c_.ma_n], axis = 1)
		df['Label'] = df.apply(lambda x : 1 if x['ma_n'] > x['Close']*(1+threshold) else (2 if x['ma_n'] < x['Close']*(1-threshold) else 0), axis = 1)
		
		df = df[['ma_n', 'Label']]

		return df


	# def parse_lab_data_3(self, df_ohlcv, threshold):

	# 	df = df_ohlcv['Close'].to_frame().reset_index(drop = True)
	# 	df['Close_'] = df.shift(-1)
		
	# 	df['Label'] = df.apply(lambda x : 1 if x['Close_'] > x['Close']*(1+threshold) else (2 if x['Close_'] < x['Close']*(1-threshold) else 0), axis = 1)
		
	# 	df = df['Label'].to_frame()

	# 	return df


	def extract_first_label(self, df):

		df_ = df.copy()

		for i in range(1, 4):

			df_[str(i)] = df_['Label'].shift(i)

	
		df_['Label'] = df_.apply(lambda x : 0 if x['1'] in [1, 2] and x['1'] == x['2'] == x['3'] else x['Label'], axis = 1) 
		
		df['Label'] = df_['Label']

		return df


	def gen_ensemble_data(self, df_inp, df_lab):
		
		df = pd.concat([df_inp, df_lab], axis = 1)
		
		df = df.dropna(how = 'any')
		
	
		delCols = ['ma_p', 'ma_n']

		for c in delCols:
			if c in df.columns:
				del df[c]

		
		df_ = df.copy()
		df_ = df_.groupby('Label')
		df_0 = None
		df_1 = None
		df_2 = None

		for name, df_group in df_:
			if name == 0:
				df_0 = df_group.copy()
			elif name == 1:
				df_1 = df_group.copy()
			elif name == 2:
				df_2 = df_group.copy()

		size = (len(df_1) + len(df_2) ) / 2 * 1
		count = int(len(df_0) // size)

		for i in range(count - 1):

			df = df.append(df_1)
			df = df.append(df_2)

		df = shuffle(df)

		return df