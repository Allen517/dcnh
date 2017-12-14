# -*- coding:utf8 -*-

import random
import math
import numpy as np
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from gcn.LogHandler import LogHandler

class _SNE(object):

    def __init__(self, graph, lr=.001, rho=[.001,.001], mode_size=1000, rep_size=128, batch_size=100
                    , table_size=1e8, negative_ratio=5, order=3, identity_labels=None):
        '''
        graph: {'src':src_graph, 'obj':obj_graph}
        '''
        if not graph or not isinstance(graph, dict):
            print 'Graph should be a dictionary set of graphs'
            return

        self.epsilon = 1e-7

        self.logger = LogHandler('sne')

        self.g = graph
        self.look_up = {key:self.g[key].look_up_dict for key in self.g.keys()}
        self.identity_labels = list(identity_labels)
        self.order = order

        self.node_size = {key:graph[key].G.number_of_nodes() for key in graph.keys()}
        self.mode_size = mode_size
        self.rep_size = rep_size
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio
        self.table_size = table_size
        
        self.lr = lr
        self.rho = rho
        self.cur_epoch = 0

        self.gen_sampling_table()

        self.sess = tf.Session()
        cur_seed = random.getrandbits(32)
        initializer = tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            self.build_graph()
        self.sess.run(tf.global_variables_initializer())

    def sigmoid(self, vec1, vec2, axis):
        return tf.nn.sigmoid(tf.reduce_sum(tf.multiply(vec1, vec2), axis=axis))

    def clip(self, vec):
        return tf.clip_by_value(vec, self.epsilon, 1-self.epsilon)

    def build_src_graph(self):
        self.pos_h_src = tf.placeholder(tf.int32, [None]) # positive cases from (h->t)
        self.pos_t_src = tf.placeholder(tf.int32, [None]) # positive cases from (t<-h)
        self.pos_h_src_v = tf.placeholder(tf.int32, [None, self.negative_ratio]) # vector of h
        self.neg_t_src = tf.placeholder(tf.int32, [None, self.negative_ratio]) # negative cases

        cur_seed = random.getrandbits(32)
        self.embeddings_src = tf.get_variable(name="embeddings_src"+str(self.order), 
                                shape=[self.node_size['src'], self.mode_size], 
                                initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
        embeddings = tf.matmul(self.embeddings_src, self.codebook, a_is_sparse=True)
        self.context_embeddings_src = tf.get_variable(name="context_embeddings_src"+str(self.order), 
                                        shape=[self.node_size['src'], self.rep_size], 
                                        initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
        self.pos_h_src_e = tf.nn.embedding_lookup(embeddings, self.pos_h_src)
        self.pos_t_src_e = tf.nn.embedding_lookup(embeddings, self.pos_t_src)
        self.pos_t_src_e_context = tf.nn.embedding_lookup(self.context_embeddings_src, self.pos_t_src)
        self.pos_h_src_v_e = tf.nn.embedding_lookup(embeddings, self.pos_h_src_v)
        self.neg_t_src_e = tf.nn.embedding_lookup(embeddings, self.neg_t_src)
        self.neg_t_src_e_context = tf.nn.embedding_lookup(self.context_embeddings_src, self.neg_t_src)


        sample_sum2 = tf.reduce_sum(
                            tf.log(
                                self.clip(1-self.sigmoid(self.pos_h_src_v_e, self.neg_t_src_e_context, 2))
                            )
                            , axis=1)
        self.second_loss_src = tf.reduce_mean(
                                    -tf.log(
                                        self.clip(self.sigmoid(self.pos_h_src_e, self.pos_t_src_e_context, 1))
                                        )
                                    -sample_sum2
                                )
        sample_sum1 = tf.reduce_sum(
                                tf.log(
                                    self.clip(1-self.sigmoid(self.pos_h_src_v_e, self.neg_t_src_e, 2))
                                )
                            , axis=1)
        self.first_loss_src = tf.reduce_mean(
                                    -tf.log(
                                        self.clip(self.sigmoid(self.pos_h_src_e, self.pos_t_src_e, 1))
                                        ) 
                                    -sample_sum1
                                )
        self.l1_loss_src = tf.reduce_mean(tf.reduce_sum(self.embeddings_src, axis=1))

    def build_obj_graph(self):
        self.pos_h_obj = tf.placeholder(tf.int32, [None]) # positive cases from (h->t)
        self.pos_t_obj = tf.placeholder(tf.int32, [None]) # positive cases from (t<-h)
        self.pos_h_obj_v = tf.placeholder(tf.int32, [None, self.negative_ratio]) # vector of h
        self.neg_t_obj = tf.placeholder(tf.int32, [None, self.negative_ratio]) # negative cases

        cur_seed = random.getrandbits(32)
        self.embeddings_obj = tf.get_variable(name="embeddings_obj"+str(self.order), 
                                shape=[self.node_size['obj'], self.mode_size], 
                                initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
        embeddings = tf.matmul(self.embeddings_obj, self.codebook, a_is_sparse=True)
        self.context_embeddings_obj = tf.get_variable(name="context_embeddings_obj"+str(self.order), 
                                        shape=[self.node_size['obj'], self.rep_size], 
                                        initializer = tf.contrib.layers.xavier_initializer(uniform = False, seed=cur_seed))
        self.pos_h_obj_e = tf.nn.embedding_lookup(embeddings, self.pos_h_obj)
        self.pos_t_obj_e = tf.nn.embedding_lookup(embeddings, self.pos_t_obj)
        self.pos_t_obj_e_context = tf.nn.embedding_lookup(self.context_embeddings_obj, self.pos_t_obj)
        self.pos_h_obj_v_e = tf.nn.embedding_lookup(embeddings, self.pos_h_obj_v)
        self.neg_t_obj_e = tf.nn.embedding_lookup(embeddings, self.neg_t_obj)
        self.neg_t_obj_e_context = tf.nn.embedding_lookup(self.context_embeddings_obj, self.neg_t_obj)

        sample_sum2 = tf.reduce_sum(
                            tf.log(
                                self.clip(1-self.sigmoid(self.pos_h_obj_v_e, self.neg_t_obj_e_context, 2))
                            )
                            , axis=1)
        self.second_loss_obj = tf.reduce_mean(
                                    -tf.log(
                                        self.clip(self.sigmoid(self.pos_h_obj_e, self.pos_t_obj_e_context, 1))
                                        )
                                    -sample_sum2
                                )
        sample_sum1 = tf.reduce_sum(
                                tf.log(
                                    self.clip(1-self.sigmoid(self.pos_h_obj_v_e, self.neg_t_obj_e, 2))
                                )
                            , axis=1)
        self.first_loss_obj = tf.reduce_mean(
                                    -tf.log(
                                        self.clip(self.sigmoid(self.pos_h_obj_e, self.pos_t_obj_e, 1))
                                        ) 
                                    -sample_sum1
                                )
        self.l1_loss_obj = tf.reduce_mean(tf.reduce_sum(self.embeddings_obj, axis=1))

    def build_aligns(self):
        identity_users_src = tuple()
        identity_users_obj = tuple()
        for u,v in self.identity_labels:
            identity_users_src += self.look_up['src'][u],
            identity_users_obj += self.look_up['obj'][v],
        align_u = tf.nn.embedding_lookup(self.embeddings_src, identity_users_src)
        align_v = tf.nn.embedding_lookup(self.embeddings_obj, identity_users_obj)

        self.align_loss = tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square(align_u-align_v), axis=1)))

    # def build_cos_aligns(self):
    #     identity_users_src = tuple()
    #     identity_users_obj = tuple()
    #     for u,v in self.identity_labels:
    #         identity_users_src += self.look_up['src'][u],
    #         identity_users_obj += self.look_up['obj'][v],
    #     align_u = tf.nn.embedding_lookup(self.embeddings_src, identity_users_src)
    #     align_v = tf.nn.embedding_lookup(self.embeddings_obj, identity_users_obj)
    #     # align_u = tf.nn.embedding_lookup(tf.clip_by_value(self.embeddings_src,0.,1.), identity_users_src)
    #     # align_v = tf.nn.embedding_lookup(tf.clip_by_value(self.embeddings_obj,0.,1.), identity_users_obj)

    #     self.align_loss = -tf.reduce_mean(tf.reduce_sum(
    #                             tf.multiply(tf.nn.l2_normalize(align_u, 1), tf.nn.l2_normalize(align_v, 1))
    #                             , axis=1))

    def build_graph(self):
        cur_seed = random.getrandbits(32)
        self.codebook = tf.get_variable(name='codebook', shape=[self.mode_size, self.rep_size],
                                        initializer=tf.contrib.layers.xavier_initializer(uniform=False, seed=cur_seed))
        self.build_src_graph()
        self.build_obj_graph()
        # self.build_cos_aligns()
        self.build_aligns()

        if self.order == 1:
            self.loss = self.first_loss_src+self.first_loss_obj\
                            +self.rho[0]*self.align_loss
                            # +self.rho[0]*(self.l1_loss_src+self.l1_loss_obj)\
        if self.order == 2:
            self.loss = self.second_loss_src+self.second_loss_obj\
                            +self.rho[0]*self.align_loss
                            # +self.rho[0]*(self.l1_loss_src+self.l1_loss_obj)\
        else:
            self.loss = self.first_loss_src+self.second_loss_src+self.first_loss_obj+self.second_loss_obj\
                            +self.rho[0]*self.align_loss
                            # +self.rho[0]*(self.l1_loss_src+self.l1_loss_obj)\
        optimizer = tf.train.AdamOptimizer(self.lr)
        self.train_op = optimizer.minimize(self.loss)

    def train_one_epoch(self):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            pos_h_src, pos_h_src_v, pos_t_src, neg_t_src, pos_h_obj, pos_h_obj_v, pos_t_obj, neg_t_obj = batch
            feed_dict = {
                self.pos_h_src : pos_h_src,
                self.pos_h_src_v : pos_h_src_v,
                self.pos_t_src : pos_t_src,
                self.neg_t_src : neg_t_src,
                self.pos_h_obj : pos_h_obj,
                self.pos_h_obj_v : pos_h_obj_v,
                self.pos_t_obj : pos_t_obj,
                self.neg_t_obj : neg_t_obj
            }
            # _, cur_loss, l1_loss_src, l1_loss_obj, align_loss = self.sess.run([self.train_op, self.loss\
            #                                         , self.l1_loss_src, self.l1_loss_obj\
            #                                         , self.align_loss],feed_dict)
            _, cur_loss, align_loss = self.sess.run([self.train_op, self.loss\
                                                    , self.align_loss],feed_dict)
            sum_loss += cur_loss
            batch_id += 1
            # self.logger.info('Finish processing batch {} and loss={}, l1_loss(src,obj)={},{}\
            #                     , align_loss={}'.format(batch_id, cur_loss, l1_loss_src, l1_loss_obj,\
            #                         align_loss))
            self.logger.info('Finish processing batch {} and loss={}, align_loss={}'
                                .format(batch_id, cur_loss, align_loss))
            # if batch_id%2==0:
            #     print 'Saving embedding in batch #{}'.format(batch_id)
            #     self.save_embeddings('tmp_embedding_{}'.format(batch_id))
        self.logger.info('epoch:{} sum of loss:{!s}'.format(self.cur_epoch, sum_loss/len(batches)))
        self.cur_epoch += 1

    def get_random_node_pairs(self, i, key, shuffle_indices, edges, edge_set, numNodes):
        if not random.random() < self.edge_prob[key][shuffle_indices[key][i]]:
            shuffle_indices[key][i] = self.edge_alias[key][shuffle_indices[key][i]]
        cur_h = edges[key][shuffle_indices[key][i]][0]
        head = cur_h*numNodes[key]
        cur_t = edges[key][shuffle_indices[key][i]][1]
        cur_h_v = []
        cur_neg_t = []
        for j in range(self.negative_ratio):
            rn = self.sampling_table[key][random.randint(0, self.table_size-1)]
            while head+rn in edge_set or cur_h == rn or rn in cur_neg_t:
                rn = self.sampling_table[key][random.randint(0, self.table_size-1)]
            cur_h_v.append(cur_h)
            cur_neg_t.append(rn)
        return cur_h, cur_h_v, cur_t, cur_neg_t

    def batch_iter(self):
        numNodes = self.node_size

        edges = {key:[(self.look_up[key][x[0]], self.look_up[key][x[1]]) for x in self.g[key].G.edges()] for key in self.g.keys()}

        data_size = {key:self.g[key].G.number_of_edges() for key in self.g.keys()}
        edge_set = {key:set([x[0]*numNodes[key]+x[1] for x in edges[key]]) for key in self.g.keys()}
        shuffle_indices = {key:np.random.permutation(np.arange(data_size[key])) for key in self.g.keys()}

        start_index = 0
        min_data_size = min(data_size['src'], data_size['obj'])
        end_index = min(start_index+self.batch_size, min_data_size)
        while start_index < min_data_size:
            pos_h_src = []
            pos_h_src_v = []
            pos_t_src = []
            neg_t_src = []
            pos_h_obj = []
            pos_h_obj_v = []
            pos_t_obj = []
            neg_t_obj = []

            for i in range(start_index, end_index):
                for key in self.g.keys():
                    cur_h, cur_h_v, cur_t, cur_neg_t = self.get_random_node_pairs(i, key
                                                        , shuffle_indices, edges, edge_set, numNodes)
                    if key == 'src':
                        pos_h_src.append(cur_h)
                        pos_h_src_v.append(cur_h_v)
                        pos_t_src.append(cur_t)
                        neg_t_src.append(cur_neg_t)
                    if key == 'obj':
                        pos_h_obj.append(cur_h)
                        pos_h_obj_v.append(cur_h_v)
                        pos_t_obj.append(cur_t)
                        neg_t_obj.append(cur_neg_t)

            yield pos_h_src, pos_h_src_v, pos_t_src, neg_t_src, pos_h_obj, pos_h_obj_v, pos_t_obj, neg_t_obj 
            start_index = end_index
            end_index = min(start_index+self.batch_size, min_data_size)

        # if end_index<data_size['src']:
        #     start_index = end_index
        #     end_index = min(start_index+self.batch_size, data_size['src'])
        #     while start_index < data_size['src']:
        #         pos_h_src = []
        #         pos_h_src_v = []
        #         pos_t_src = []
        #         neg_t_src = []
        #         pos_h_obj = []
        #         pos_h_obj_v = []
        #         pos_t_obj = []
        #         neg_t_obj = []

        #         for i in range(start_index, end_index):
        #             cur_h, cur_h_v, cur_t, cur_neg_t =self.get_random_node_pairs(i, 'src'
        #                                                 , shuffle_indices, edges, edge_set, numNodes)
        #             pos_h_src.append(cur_h)
        #             pos_h_src_v.append(cur_h_v)
        #             pos_t_src.append(cur_t)
        #             neg_t_src.append(cur_neg_t)

        #         yield pos_h_src, pos_h_src_v, pos_t_src, neg_t_src, pos_h_obj, pos_h_obj_v, pos_t_obj, neg_t_obj 
        #         start_index = end_index
        #         end_index = min(start_index+self.batch_size, data_size['src'])
        # else:
        #     start_index = end_index
        #     end_index = min(start_index+self.batch_size, data_size['obj'])
        #     while start_index < data_size['obj']:
        #         pos_h_src = []
        #         pos_h_src_v = []
        #         pos_t_src = []
        #         neg_t_src = []
        #         pos_h_obj = []
        #         pos_h_obj_v = []
        #         pos_t_obj = []
        #         neg_t_obj = []

        #         for i in range(start_index, end_index):
        #             cur_h, cur_h_v, cur_t, cur_neg_t = self.get_random_node_pairs(i, 'obj'
        #                                                 , shuffle_indices, edges, edge_set, numNodes)
        #             pos_h_obj.append(cur_h)
        #             pos_h_obj_v.append(cur_h_v)
        #             pos_t_obj.append(cur_t)
        #             neg_t_obj.append(cur_neg_t)

        #         yield pos_h_src, pos_h_src_v, pos_t_src, neg_t_src, pos_h_obj, pos_h_obj_v, pos_t_obj, neg_t_obj 
        #         start_index = end_index
        #         end_index = min(start_index+self.batch_size, data_size['obj'])

    def gen_sampling_table(self):
        power = 0.75
        numNodes = self.node_size

        self.logger.info("Pre-procesing for non-uniform negative sampling!")
        node_degree = {key:np.zeros(numNodes[key]) for key in self.g.keys()} # out degree

        self.look_up = {key:self.g[key].look_up_dict for key in self.g.keys()}
        for key in self.g.keys():
            for edge in self.g[key].G.edges():
                node_degree[key][self.look_up[key][edge[0]]] += self.g[key].G[edge[0]][edge[1]]["weight"]

        norm = {key:sum([math.pow(node_degree[key][i], power) for i in range(numNodes[key])]) for key in self.g.keys()}

        self.sampling_table = {key:np.zeros(int(self.table_size), dtype=np.uint32) for key in self.g.keys()}

        for key in self.g.keys():
            p = 0
            i = 0
            for j in range(numNodes[key]):
                p += float(math.pow(node_degree[key][j], power)) / norm[key]
                while i < self.table_size and float(i) / self.table_size < p:
                    self.sampling_table[key][i] = j
                    i += 1

        data_size = {key:self.g[key].G.number_of_edges() for key in self.g.keys()}
        self.edge_alias = {key:np.zeros(data_size[key], dtype=np.int32) for key in self.g.keys()}
        self.edge_prob = {key:np.zeros(data_size[key], dtype=np.float32) for key in self.g.keys()}
        large_block = {key:np.zeros(data_size[key], dtype=np.int32) for key in self.g.keys()}
        small_block = {key:np.zeros(data_size[key], dtype=np.int32) for key in self.g.keys()}

        for key in self.g.keys():
            total_sum = sum([self.g[key].G[edge[0]][edge[1]]["weight"] for edge in self.g[key].G.edges()])
            norm_prob = [self.g[key].G[edge[0]][edge[1]]["weight"]*data_size[key]/total_sum 
                                for edge in self.g[key].G.edges()]

            num_small_block = 0
            num_large_block = 0
            cur_small_block = 0
            cur_large_block = 0
            for k in range(data_size[key]-1, -1, -1):
                if norm_prob[k] < 1:
                    small_block[key][num_small_block] = k
                    num_small_block += 1
                else:
                    large_block[key][num_large_block] = k
                    num_large_block += 1
            while num_small_block and num_large_block:
                num_small_block -= 1
                cur_small_block = small_block[key][num_small_block]
                num_large_block -= 1
                cur_large_block = large_block[key][num_large_block]
                self.edge_prob[key][cur_small_block] = norm_prob[cur_small_block]
                self.edge_alias[key][cur_small_block] = cur_large_block
                norm_prob[cur_large_block] = norm_prob[cur_large_block] + norm_prob[cur_small_block] -1
                if norm_prob[cur_large_block] < 1:
                    small_block[key][num_small_block] = cur_large_block
                    num_small_block += 1
                else:
                    large_block[key][num_large_block] = cur_large_block
                    num_large_block += 1

            while num_large_block:
                num_large_block -= 1
                self.edge_prob[key][large_block[key][num_large_block]] = 1
            while num_small_block:
                num_small_block -= 1
                self.edge_prob[key][small_block[key][num_small_block]] = 1

    def get_one_embeddings(self, key, tf_embeddings):
        vectors = dict()
        embeddings = tf_embeddings.eval(session=self.sess)
        look_back = self.g[key].look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors

    def get_codebook(self):
        vectors = dict()
        codebook = self.codebook.eval(session=self.sess)
        for i, cb in enumerate(codebook):
            vectors[i] = cb
        return vectors

    def get_embeddings(self):
        vectors_src = self.get_one_embeddings('src', self.embeddings_src)
        vectors_obj = self.get_one_embeddings('obj', self.embeddings_obj)
        codebook = self.get_codebook()
        return {'src':vectors_src, 'obj':vectors_obj, 'codebook':codebook}


class SNE(object):

    def __init__(self, graph, lr=.001, rho=[.001,.001], mode_size=1000, rep_size=128, batch_size=1000, epoch=10
                    ,table_size=1e8, negative_ratio=5, order=3, label_file = None, auto_stop = True):
        if not graph or not isinstance(graph, dict):
            print 'Param graph should be a dictionary set of graphs'
            return
        self.mode_size = mode_size
        self.rep_size = rep_size
        self.order = order
        self.best_result = 0
        self.vectors = {}
        if label_file:
            identity_labels = self.read_identity_labels(filename=label_file)
            self.model = _SNE(graph, lr=lr, rho=rho, mode_size=mode_size, rep_size=rep_size
                                , batch_size=batch_size, table_size=table_size, negative_ratio=negative_ratio
                                , order=self.order, identity_labels=identity_labels)
            for i in range(epoch):
                    self.model.train_one_epoch()
                    if i%2==0:
                        self.get_embeddings()
                        self.save_embeddings('tmp_embedding_{}'.format(i))
            self.get_embeddings()

    def read_identity_labels(self, filename):
        with open(filename, 'r') as fin:
            for line in fin:
                ln = line.strip()
                if not ln:
                    break
                yield ln.split()

    def get_embeddings(self):
        self.last_vectors = self.vectors
        self.vectors = self.model.get_embeddings()

    def save_embeddings(self, filename):
        for key in self.vectors.keys():
            fout = open(filename+'-'+key, 'w')
            node_num = len(self.vectors[key].keys())
            fout.write("{} {}\n".format(node_num, self.rep_size))
            for node, vec in self.vectors[key].items():
                fout.write("{} {}\n".format(node,
                                            ' '.join([str(x) for x in vec])))
            fout.close()