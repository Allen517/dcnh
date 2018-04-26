# -*- coding:utf8 -*-

import random
import tensorflow as tf
import numpy as np
import os,sys

from utils.LogHandler import LogHandler
from utils.utils import load_train_valid_labels, batch_iter, valid_iter

class DCNH_SP(object):

	def __init__(self, learning_rate, batch_size, neg_ratio, n_input, n_out, n_hidden, n_layer
					, device, files, log_file):
		if os.path.exists('log/'+log_file+'.log'):
			os.remove('log/'+log_file+'.log')
		self.logger = LogHandler(log_file)

		self.device = device

		# Parameters
		self.learning_rate = learning_rate
		self.batch_size = batch_size
		self.neg_ratio = neg_ratio
		self.valid_prop = .9
		self.valid_sample_size = 9

		self.gamma = 1
		self.eta = 0

		self.cur_epoch = 1

		# Network Parameters
		self.n_hidden = n_hidden # number of neurons in hidden layer
		self.n_input = n_input # size of node embeddings
		self.n_out = n_out # hashing code
		self.n_layer = n_layer # number of layer

		# Set Train Data
		if not isinstance(files, list) and len(files)<3:
			self.logger.info('The alogrihtm needs files like [First Graph File, Second Graph File, Label File]')
			return

		# tf Graph input
		self.lookup_f = dict()
		self.lookup_g = dict()
		self.look_back_f = list()
		self.look_back_g = list()
		self._read_train_dat(files[0], files[1], files[2]) # douban, weibo, label files
		self.valid_sample_size = min(min(self.valid_sample_size, len(self.look_back_f)-1), len(self.look_back_g)-1)

		# TF Graph Building
		self.sess = tf.Session()
		cur_seed = random.getrandbits(32)
		initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)
		with tf.device(self.device):
			with tf.variable_scope("model", reuse=None, initializer=initializer):
				self.mlp_weights()
				self.build_graph()
				self.build_valid_graph()
			self.sess.run(tf.global_variables_initializer())

	def _read_embeddings(self, embed_file, lookup, look_back):
		embedding = list()
		with open(embed_file, 'r') as emb_handler:
			idx = 0
			for ln in emb_handler:
				ln = ln.strip()
				if ln:
					elems = ln.split()
					if len(elems)==2:
						continue
					embedding.append(map(float, elems[1:]))
					lookup[elems[0]] = idx
					look_back.append(elems[0])
					idx += 1
		return np.array(embedding), lookup, look_back

	def _read_train_dat(self, embed1_file, embed2_file, label_file):
		self.L = load_train_valid_labels(label_file, self.valid_prop)
		self.F, self.lookup_f, self.look_back_f = self._read_embeddings(embed1_file, self.lookup_f, self.look_back_f)
		self.G, self.lookup_g, self.look_back_g = self._read_embeddings(embed2_file, self.lookup_g, self.look_back_g)

	def mlp_weights(self):
		# Store layers weight & bias
		self.weights = dict()
		self.biases = dict()
		self.weights['h0_f'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]))
		self.weights['h0_g'] = tf.Variable(tf.random_normal([self.n_input, self.n_hidden]))
		self.biases['b0_f'] = tf.Variable(tf.zeros([self.n_hidden]))
		self.biases['b0_g'] = tf.Variable(tf.zeros([self.n_hidden]))
 		for i in range(1,self.n_layer):
			self.weights['h{}'.format(i)] = tf.Variable(tf.random_normal([self.n_hidden, self.n_hidden]))
			self.biases['b{}'.format(i)] = tf.Variable(tf.zeros([self.n_hidden]))
		self.weights['out'] = tf.Variable(tf.random_normal([self.n_hidden, self.n_out]))
		self.biases['b_out'] = tf.Variable(tf.zeros([self.n_out]))

	def build_code_graph(self, inputs, tag):

		# Input layer
		layer = tf.nn.sigmoid(tf.add(tf.matmul(tf.reshape(inputs,[-1,self.n_input]), self.weights['h0_'+tag])
									, self.biases['b0_'+tag]))
		for i in range(1,self.n_layer):
			layer = tf.nn.sigmoid(tf.add(tf.matmul(layer, self.weights['h{}'.format(i)])
									, self.biases['b{}'.format(i)]))
		# Output fully connected layer with a neuron
		code = tf.nn.tanh(tf.matmul(layer, self.weights['out']) + self.biases['b_out'])

		return code

	def build_lin_code_graph(self, inputs, tag):

		# Output fully connected layer with a neuron
		code = tf.matmul(tf.reshape(inputs,[-1,self.n_input]), self.weights['out']) + self.biases['b_out']

		return code

	def build_train_graph(self, src_tag, obj_tag):

		PF = self.build_code_graph(self.pos_src_inputs, src_tag) # batch_size*n_out
		PG = self.build_code_graph(self.pos_obj_inputs, obj_tag) # batch_size*n_out
		NF = tf.reshape(
				self.build_code_graph(self.neg_src_inputs, src_tag)
				, [-1, self.neg_ratio, self.n_out]
				) # batch_size*neg_ratio*n_out
		NG = tf.reshape(
				self.build_code_graph(self.neg_obj_inputs, obj_tag)
				, [-1, self.neg_ratio, self.n_out]
				) # batch_size*neg_ratio*n_out
		B = tf.sign(PF+PG) # batch_size*n_out
		# self.ph['B'] = tf.sign(self.ph['F']+self.ph['G']) # batch_size*n_out

		# train loss
		term1_first = tf.log(tf.nn.sigmoid(tf.reduce_sum(.5*tf.multiply(PF, PG),axis=1)))
		term1_second = tf.reduce_sum(tf.log(1-tf.nn.sigmoid(tf.reduce_sum(.5*tf.multiply(NF, NG),axis=2))),axis=1)
		term1 = -tf.reduce_sum(term1_first+term1_second)
		term2 = tf.reduce_sum(tf.pow((B-PF),2))+tf.reduce_sum(tf.pow((B-PG),2))
		term3 = tf.reduce_sum(tf.reduce_sum(tf.pow(PF,2))+tf.reduce_sum(tf.pow(PG,2), axis=1))
		# term1 = -tf.reduce_sum(tf.multiply(self.ph['S'], theta)-tf.log(1+tf.exp(theta)))
		# term2 = tf.reduce_sum(tf.norm(self.ph['B']-self.ph['F'],axis=1))+tf.reduce_sum(tf.norm(self.ph['B']-self.ph['G'],axis=1))
		# term3 = tf.reduce_sum(tf.norm(self.ph['F'],axis=1))+tf.reduce_sum(tf.norm(self.ph['G'],axis=1))

		return (term1+self.gamma*term2+self.eta*term3)/self.cur_batch_size

	def build_graph(self):
		self.cur_batch_size = tf.placeholder('float32', name='batch_size')

		self.pos_src_inputs = tf.placeholder('float32', [None, self.n_input])
		self.pos_obj_inputs = tf.placeholder('float32', [None, self.n_input])
		self.neg_src_inputs = tf.placeholder('float32', [None, self.neg_ratio, self.n_input])
		self.neg_obj_inputs = tf.placeholder('float32', [None, self.neg_ratio, self.n_input])

		self.loss_f2g = self.build_train_graph('f', 'g')
		self.loss_g2f = self.build_train_graph('g', 'f')
		# self.loss = (term1+self.eta*term3)/self.cur_batch_size
		optimizer = tf.train.AdamOptimizer(self.learning_rate)
		self.train_op_f2g = optimizer.minimize(self.loss_f2g)
		self.train_op_g2f = optimizer.minimize(self.loss_g2f)

	def build_valid_graph(self):

		# validation
		self.valid_f_inputs = tf.placeholder('float32', [None, self.valid_sample_size, self.n_input])
		self.valid_g_inputs = tf.placeholder('float32', [None, self.valid_sample_size, self.n_input])

		valid_f = tf.reshape(
				self.build_code_graph(self.valid_f_inputs, 'f')
				, [-1, self.valid_sample_size, self.n_out]
				) # batch_size*neg_ratio*n_out
		valid_g = tf.reshape(
				self.build_code_graph(self.valid_g_inputs, 'g')
				, [-1, self.valid_sample_size, self.n_out]
				) # batch_size*neg_ratio*n_out
		# self.dot_dist = tf.reduce_sum(tf.multiply(valid_f, valid_g),axis=2)
		self.hamming_dist = -tf.reduce_sum(
								tf.clip_by_value(tf.sign(tf.multiply(valid_f,valid_g)),-1.,0.)
									, axis=2
								)

	def train_one_epoch(self):
		sum_loss = 0.0

		# train process
		# print 'start training...'
		batches_f2g = list(batch_iter(self.L, self.batch_size, self.neg_ratio\
										, self.lookup_f, self.lookup_g, 'f', 'g'))
		batches_g2f = list(batch_iter(self.L, self.batch_size, self.neg_ratio\
										, self.lookup_g, self.lookup_f, 'g', 'f'))
		n_batches = min(len(batches_f2g), len(batches_g2f))
		batch_id = 0
		for i in range(n_batches):
			# training the process from network f to network g
			pos_src_f2g,pos_obj_f2g,neg_src_f2g,neg_obj_f2g = batches_f2g[i]
			if not len(pos_src_f2g)==len(pos_obj_f2g) and not len(neg_src_f2g)==len(neg_obj_f2g):
				self.logger.info('The input label file goes wrong as the file format.')
				continue
			batch_size_f2g = len(pos_src_f2g)
			feed_dict = {
				self.pos_src_inputs:self.F[pos_src_f2g,:],
				self.pos_obj_inputs:self.G[pos_obj_f2g,:],
				self.neg_src_inputs:self.F[neg_src_f2g,:],
				self.neg_obj_inputs:self.G[neg_obj_f2g,:],
				self.cur_batch_size:batch_size_f2g
			}
			_, cur_loss_f2g = self.sess.run([self.train_op_f2g, self.loss_f2g],feed_dict)

			sum_loss += cur_loss_f2g

			# training the process from network g to network f
			pos_src_g2f,pos_obj_g2f,neg_src_g2f,neg_obj_g2f = batches_g2f[i]
			if not len(pos_src_g2f)==len(pos_obj_g2f) and not len(neg_src_g2f)==len(neg_obj_g2f):
				self.logger.info('The input label file goes wrong as the file format.')
				continue
			batch_size_g2f = len(pos_src_g2f)
			feed_dict = {
				self.pos_src_inputs:self.G[pos_src_g2f,:],
				self.pos_obj_inputs:self.F[pos_obj_g2f,:],
				self.neg_src_inputs:self.G[neg_src_g2f,:],
				self.neg_obj_inputs:self.F[neg_obj_g2f,:],
				self.cur_batch_size:batch_size_g2f
			}
			_, cur_loss_g2f = self.sess.run([self.train_op_g2f, self.loss_g2f],feed_dict)

			sum_loss += cur_loss_g2f

			batch_id += 1
			break

		# valid process
		valid_f, valid_g = valid_iter(self.L, self.valid_sample_size, self.lookup_f, self.lookup_g, 'f', 'g')
		# print valid_f,valid_g
		if not len(valid_f)==len(valid_g):
			self.logger.info('The input label file goes wrong as the file format.')
			return
		valid_size = len(valid_f)
		# for vec in valid_f:
		# 	print len(vec)
		# print self.F[valid_f,:]
		feed_dict = {
			self.valid_f_inputs:self.F[valid_f,:],
			self.valid_g_inputs:self.G[valid_g,:],
		}
		# valid_dist = self.sess.run(self.dot_dist,feed_dict)
		valid_dist = self.sess.run(self.hamming_dist,feed_dict)
		mrr = .0
		for i in range(valid_size):
			fst_dist = valid_dist[i][0]
			pos = 1
			for k in range(1,len(valid_dist[i])):
				if fst_dist>=valid_dist[i][k]:
					pos+=1
			# print pos
			# self.logger.info('dist:{},pos:{}'.format(fst_dist,pos))
			# print valid_dist[i]
			mrr += 1./pos
		self.logger.info('Epoch={}, sum of loss={!s}, mrr={}'
							.format(self.cur_epoch, sum_loss/batch_id/2, mrr/valid_size))
		# print 'mrr:',mrr/valid_size
		# self.logger.info('Epoch={}, sum of loss={!s}, valid_loss={}'
		#                     .format(self.cur_epoch, sum_loss/batch_id, valid_loss))
		self.cur_epoch += 1

	def _write_in_file(self, filename, vec, tag):
		with open(filename, 'aw') as res_handler:
			if len(vec.shape)>1:
				column_size = vec.shape[1]
			else:
				column_size = 1
			reshape_vec = vec.reshape(-1)
			vec_size = len(reshape_vec)
			res_handler.write(tag+'\n')
			for i in range(0,vec_size,column_size):
				res_handler.write('{}\n'.format(' '.join([str(reshape_vec[i+k]) for k in range(column_size)])))

	def save_models(self, filename):
		if os.path.exists(filename):
			os.remove(filename)
		for k,v in self.weights.iteritems():
			self._write_in_file(filename, v.eval(self.sess), k)
		for k,v in self.biases.iteritems():
			self._write_in_file(filename, v.eval(self.sess), k)

if __name__ == '__main__':
	res_file = 'res_file'

	# SAVING_STEP = 1
	# MAF_EPOCHS = 21
	# model = DCNH(learning_rate=0.1, batch_size=4, neg_ratio=3, n_input=4, n_out=2, n_hidden=3
	# 				,files=['tmp_res.node_embeddings_src', 'tmp_res.node_embeddings_obj', 'data/test.align'])
	SAVING_STEP = 10
	MAF_EPOCHS = 20001
	model = DCNH_SP(learning_rate=0.01, batch_size=128, neg_ratio=5, n_input=256, n_out=32, n_hidden=32, n_layer=2
					,files=['douban_all.txt', 'weibo_all.txt', 'douban_weibo.identity.users.final.p0dot8']
					,log_file='DCNH_SP'
					,device=':/gpu:0')
	for i in range(MAF_EPOCHS):
		model.train_one_epoch()
		if i>0 and i%SAVING_STEP==0:
			model.save_models(res_file+'.epoch_'+str(i))