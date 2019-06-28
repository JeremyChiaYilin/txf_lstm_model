

import tensorflow as tf
from tensorflow.contrib import rnn

import os

TRAIN_PROB = 0.5
PREDICT_PROB = 1.0

class LSTM(object):

	def __init__(self, X_DIM, Y_DIM, CELL_SIZE, STEPS,
				reuse, isRestore = False, modelPath = None):

		config = tf.ConfigProto()
		config.gpu_options.per_process_gpu_memory_fraction = 0.1
		self.sess = tf.Session(config = config)

		self.CELL_SIZE = CELL_SIZE
		self.STEPS = STEPS
		self.X_DIM = X_DIM 
		self.Y_DIM = Y_DIM

		self.X = tf.placeholder(tf.float32, [None, self.X_DIM], name = 'X')
		self.Y = tf.placeholder(tf.float32, [None, self.Y_DIM], name = 'Y')
		self.prob = tf.placeholder_with_default(1.0, shape=())


		#####################################################################
		###  not reuse in variable for older model

		with tf.variable_scope('BiRNN', reuse = reuse):
			self.output = self.build_model_1(self.X, 'BiRNN',
											tf.nn.tanh,
											#tf.truncated_normal_initializer,
											tf.contrib.layers.xavier_initializer(),
											#tf.contrib.layers.variance_scaling_initializer(),
											#tf.orthogonal_initializer(),
											rnn_layer = 2)

			# self.output = self.build_model_2(self.X, 'MultiRNN',
			# 								tf.nn.tanh,
			# 								tf.truncated_normal_initializer,
			# 								#tf.contrib.layers.xavier_initializer(),
			# 								#tf.contrib.layers.variance_scaling_initializer(),
			# 								rnn_layer = 10)


		self.loss, self.train_op = self.train_method_1(self.Y, self.output)
		self.softmax_out, self.labelIndex, self.predictIndex, self.accuracy = self.evaluate_model(self.Y,
																								self.output)
		#####################################################################

	
		self.saver = tf.train.Saver()
		if isRestore:
			self.saver.restore(self.sess, modelPath)
		else:
			self.sess.run(tf.global_variables_initializer())

	
	def predicting(self, inp):

		predict = self.sess.run(self.predictIndex, feed_dict = {self.X : inp, self.prob : PREDICT_PROB})
		prob = self.sess.run(self.softmax_out, feed_dict = {self.X : inp, self.prob : PREDICT_PROB})

		return predict, prob

	def testing(self, inp_datas, lab_datas):
		batches = len(inp_datas)
		labels = []
		predicts = []
		for i in range(batches):

			inp = inp_datas[i]
			lab = lab_datas[i]
		
			lab = self.sess.run(tf.one_hot(indices = lab, depth = self.Y_DIM,
												on_value = 1.0, off_value = 0.0, axis = -1))


			loss = self.sess.run(self.loss,
								feed_dict = {self.X : inp,
											self.Y : lab,
											self.prob : PREDICT_PROB})

			accuracy = self.sess.run(self.accuracy, feed_dict = {self.X : inp,
																self.Y : lab,
																self.prob : PREDICT_PROB})

			label = self.sess.run(self.labelIndex, feed_dict = {self.Y : lab, self.prob : PREDICT_PROB})

			predict = self.sess.run(self.predictIndex, feed_dict = {self.X : inp, self.prob : PREDICT_PROB})

			precision = self.analyzeResult(label, predict)

			

			print('testing - step : {} , loss {}'.format(i, loss))

			print('testing - step : ', i ,
				', accuracy : ', accuracy,
				', precision : ', precision,
				', label : ', label,
				', predict : ', predict)


			#output = self.sess.run(self.softmax_out, feed_dict = {self.X : inp})
			#print(output)

			labels.append(label)
			predicts.append(predict)

		return loss, accuracy, precision

	def training(self, inp_datas, lab_datas):

		batches = len(inp_datas)
		for i in range(batches):	
			inp = inp_datas[i]
			lab = lab_datas[i]
			lab = self.sess.run(tf.one_hot(indices = lab,
											depth = self.Y_DIM,
											on_value = 1.0,
											off_value = 0.0,
											axis = -1))

			self.sess.run(self.train_op,
							feed_dict = {self.X : inp,
										self.Y : lab,
										self.prob : TRAIN_PROB})

			if i % 100 == 0:
				loss = self.sess.run(self.loss,
									feed_dict = {self.X : inp,
												self.Y : lab,
												self.prob : TRAIN_PROB})
				print('step : {} , loss {}'.format(i, loss))

				accuracy = self.sess.run(self.accuracy, feed_dict = {self.X : inp,
																	self.Y : lab,
																	self.prob : TRAIN_PROB})

				label = self.sess.run(self.labelIndex, feed_dict = {self.Y : lab, self.prob : TRAIN_PROB})

				predict = self.sess.run(self.predictIndex, feed_dict = {self.X : inp, self.prob : TRAIN_PROB})

				precision = self.analyzeResult(label, predict)

				print('step : ', i , ' , accuracy : ', accuracy, ', precision : ', precision,
					', label : ', label, ', predict : ', predict)


	def build_model_1(self, X, type, activation, initializer, rnn_layer = 2, trainable = True):

		if type == 'BiRNN':
			weight = tf.get_variable('rnn_w',
									[2 * self.CELL_SIZE, self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			bias = tf.get_variable('rnn_b',
									[self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			rnn, _ = self.BiRNN(X, weight, bias, initializer, activation)
		elif type == 'MultiRNN':
			weight = tf.get_variable('rnn_w',
									 [self.CELL_SIZE, self.Y_DIM],
									 trainable = trainable,
									 initializer = initializer)
			bias = tf.get_variable('rnn_b',
								[self.Y_DIM],
								trainable = trainable,
								initializer = initializer)

			rnn, _ = self.MultiRNN(X, weight, bias, initializer, activation, rnn_layer)	

		
		return rnn


	## RNN + Fully Connection
	def build_model_2(self, X, type, activation, initializer, rnn_layer = 2, trainable = True):

		if type == 'BiRNN':
			CellSize = 2 * self.CELL_SIZE
			weight = tf.get_variable('rnn_w',
									[CellSize, self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			bias = tf.get_variable('rnn_b',
									[self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			rnn, outputs = self.BiRNN(X, weight, bias, initializer, activation)

		elif type == 'MultiRNN':
			CellSize = self.CELL_SIZE
			weight = tf.get_variable('rnn_w',
									 [CellSize, self.Y_DIM],
									 trainable = trainable,
									 initializer = initializer)
			bias = tf.get_variable('rnn_b',
								[self.Y_DIM],
								trainable = trainable,
								initializer = initializer)
			rnn, outputs = self.MultiRNN(X, weight, bias, initializer, activation, rnn_layer)	



		fc_node = [20, 20, 20, 10, 10, 10, 5, 5, 5]

		dense = tf.layers.dense(inputs = outputs[-1], units = 20, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)

		for n in fc_node:

			dense = tf.layers.dense(inputs = dense, units = n, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)


		
		out = tf.layers.dense(inputs = dense, units = self.Y_DIM, activation = activation,
		 					kernel_initializer = initializer,
		 					bias_initializer = initializer)

		# n_l1 = 20
		# n_l2 = 10
		# n_l3 = self.Y_DIM

		
		# dense1 = tf.layers.dense(inputs = outputs[-1], units = n_l1, activation = activation,
		# 						kernel_initializer = initializer,
		# 						bias_initializer = initializer)

		# dense2 = tf.layers.dense(inputs = dense1, units = n_l2, activation = activation,
		# 						kernel_initializer = initializer,
		# 						bias_initializer = initializer)

		# out = tf.layers.dense(inputs = dense2, units = n_l3, activation = activation,
		# 						kernel_initializer = initializer,
		# 						bias_initializer = initializer)



		return out


	## FC + RNN + FC
	def build_model_3(self, X, type, activation, initializer, rnn_layer = 2, trainable = True):

		n_l1 = round(self.X_DIM * 1.8)
		n_l2 = round(self.X_DIM * 1.2)
		n_l3 = self.X_DIM

		
		dense1 = tf.layers.dense(inputs = X, units = n_l1, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)

		dense2 = tf.layers.dense(inputs = dense1, units = n_l2, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)

		dense3 = tf.layers.dense(inputs = dense2, units = n_l3, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)



		if type == 'BiRNN':
			CellSize = 2 * self.CELL_SIZE
			weight = tf.get_variable('rnn_w',
									[CellSize, self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			bias = tf.get_variable('rnn_b',
									[self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			rnn, outputs = self.BiRNN(dense3, weight, bias, initializer, activation)
		elif type == 'MultiRNN':
			CellSize = self.CELL_SIZE
			weight = tf.get_variable('rnn_w',
									 [CellSize, self.Y_DIM],
									 trainable = trainable,
									 initializer = initializer)
			bias = tf.get_variable('rnn_b',
								[self.Y_DIM],
								trainable = trainable,
								initializer = initializer)
			rnn, outputs = self.MultiRNN(dense3, weight, bias, initializer, activation, rnn_layer)	

	
		n_l1 = 20
		n_l2 = 10
		n_l3 = self.Y_DIM

		
		dense4 = tf.layers.dense(inputs = outputs[-1], units = n_l1, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)

		dense5 = tf.layers.dense(inputs = dense4, units = n_l2, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)

		out = tf.layers.dense(inputs = dense5, units = n_l3, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)



		return out

	## pooling
	def build_model_4(self, X, type, activation, initializer, rnn_layer = 2, trainable = True):

		if type == 'BiRNN':
			weight = tf.get_variable('rnn_w',
									[2 * self.CELL_SIZE, self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			bias = tf.get_variable('rnn_b',
									[self.Y_DIM],
									trainable = trainable,
									initializer = initializer)
			rnn, outputs = self.BiRNN(X, weight, bias, initializer, activation)
		elif type == 'MultiRNN':
			weight = tf.get_variable('rnn_w',
									 [self.CELL_SIZE, self.Y_DIM],
									 trainable = trainable,
									 initializer = initializer)
			bias = tf.get_variable('rnn_b',
								[self.Y_DIM],
								trainable = trainable,
								initializer = initializer)

			rnn, outputs = self.MultiRNN(X, weight, bias, initializer, activation, rnn_layer)	

		# o = []
		# for i in [-3, -2, -1]:
		# 	dense = tf.layers.dense(inputs = outputs[i], units = 8, activation = activation,
		# 							kernel_initializer = initializer,
		# 							bias_initializer = initializer)

		# 	fc_node = [8, 5, 5, 3, 3]
		# 	for n in fc_node:

		# 		dense = tf.layers.dense(inputs = dense, units = n, activation = activation,
		# 							kernel_initializer = initializer,
		# 							bias_initializer = initializer)

		# 	o.append(dense)


		# output = tf.reshape(o, [-1, 3 * 3])

		# out = tf.layers.dense(inputs = output, units = self.Y_DIM, activation = activation,
		# 						kernel_initializer = initializer,
		# 						bias_initializer = initializer)


		## FC
		# output = tf.reshape(outputs[0:5], [-1, 5 * 16])

		# dense = tf.layers.dense(inputs = output, units = 48, activation = activation,
		# 						kernel_initializer = initializer,
		# 						bias_initializer = initializer)
		# fc_node = [48, 48, 48, 30, 30, 30, 30, 18, 18, 18, 18, 18, 7, 7, 7, 7, 7, 7, 3, 3, 3, 3, 3, 3]
		# for n in fc_node:

		# 	dense = tf.layers.dense(inputs = dense, units = n, activation = activation,
		# 						kernel_initializer = initializer,
		# 						bias_initializer = initializer)


		
		# out = tf.layers.dense(inputs = dense, units = self.Y_DIM, activation = activation,
		#  					kernel_initializer = initializer,
		#  					bias_initializer = initializer)
		

		## pooling
		output = tf.expand_dims(outputs[-1], -1)	
		pool = tf.layers.max_pooling1d(inputs = output, pool_size = [2], strides = 1)
		pool = tf.reshape(pool, [-1, 15])
		out = tf.layers.dense(inputs = pool, units = self.Y_DIM, activation = activation,
								kernel_initializer = initializer,
								bias_initializer = initializer)

		return out

	
	def train_method_1(self, label, prediction):


		loss = tf.losses.softmax_cross_entropy(onehot_labels = label, logits = prediction)
		op = tf.train.AdamOptimizer().minimize(loss)

		# tv = tf.trainable_variables()
		# regularization_cost = 0.001* tf.reduce_sum([ tf.nn.l2_loss(v) for v in tv ]) 
		# loss = tf.losses.softmax_cross_entropy(onehot_labels = label, logits = prediction)
		# loss = loss + regularization_cost
		
		# op = tf.train.AdamOptimizer().minimize(loss)



		return loss, op

	def BiRNN(self, x, weight, bias, initializer, activation):

		x = tf.unstack(tf.reshape(x, [tf.shape(x)[0], self.STEPS, int(self.X_DIM / self.STEPS)]), self.STEPS, axis = 1)
	
		# lstm_fw_cell = rnn.CoupledInputForgetGateLSTMCell(self.CELL_SIZE,
		# 													forget_bias=10.0, 
		# 													use_peepholes = True, 
		# 													proj_clip = 15.0,
		# 													initializer = initializer,
		# 													activation = activation)
		# lstm_bw_cell = rnn.CoupledInputForgetGateLSTMCell(self.CELL_SIZE,
		# 													forget_bias=10.0,
		# 													use_peepholes = True,
		# 													proj_clip = 15.0,
		# 													initializer = initializer,
		# 													activation = activation)

		lstm_fw_cell = rnn.LayerNormBasicLSTMCell(self.CELL_SIZE,
												forget_bias=10.0,
												dropout_keep_prob = self.prob,
												activation = activation)
		lstm_bw_cell = rnn.LayerNormBasicLSTMCell(self.CELL_SIZE,
												forget_bias=10.0,
												dropout_keep_prob = self.prob,
												activation = activation)


		# lstm_fw_cell = rnn.GRUCell(self.CELL_SIZE, activation = tf.nn.relu)
		# lstm_bw_cell = rnn.GRUCell(self.CELL_SIZE, activation = tf.nn.relu)

		outputs, _, _ = tf.nn.static_bidirectional_rnn(lstm_fw_cell,
														lstm_bw_cell,
														x,
														dtype=tf.float32)

		#out = tf.layers.batch_normalization(outputs[-1])
		
		return tf.matmul(outputs[-1], weight) + bias, outputs

		# output = tf.nn.tanh(tf.matmul(outputs[-1], weight) + bias)
		# return output, outputs


	def MultiRNN(self, x, weight, bias, initializer, activation, num_layers):
		
		x = tf.unstack(tf.reshape(x, [tf.shape(x)[0], self.STEPS, int(self.X_DIM / self.STEPS)]), self.STEPS, axis = 1)
		lstm_cell = rnn.CoupledInputForgetGateLSTMCell(self.CELL_SIZE, 
															forget_bias=10.0, 
															use_peepholes = True, 
															proj_clip = 15.0,
															initializer = initializer,
															activation = activation)
		lstm_cell = rnn.MultiRNNCell([lstm_cell] * num_layers)
		outputs, _= rnn.static_rnn(lstm_cell, x , dtype = tf.float32)

		return tf.matmul(outputs[-1], weight) + bias, outputs

	def evaluate_model(self, label, prediction): 

		softmax = tf.nn.softmax(prediction)

		labIndex = tf.argmax(label, axis = 1)
		preIndex = tf.argmax(prediction, axis = 1)

		correct = tf.equal(labIndex, preIndex)
		accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

		return softmax, labIndex, preIndex, accuracy

	def analyzeResult(self, label, predict):

		size = len(label)

		tp = 0
		fp = 0
	
		for i in range(size):
			lab = label[i]
			pre = predict[i]

			if pre == 1:
				if lab == 1:
					tp += 1
				elif lab == 2:
					fp += 1
			elif pre == 2:
				if lab == 2:
					tp += 1
				elif lab == 1:
					fp += 1

		if (tp + fp) != 0:
			precision = tp / (tp + fp)
		else:
			precision = 0

		return precision

	def save_model(self, _dir, fileName):

		if not (os.path.exists(_dir)):
				os.makedirs(_dir)
		self.saver.save(self.sess, _dir + fileName)


	def close(self):
		self.sess.close()
		tf.reset_default_graph()