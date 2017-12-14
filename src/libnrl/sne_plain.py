# -*- coding:utf8 -*-

import random
import math
import numpy as np
from collections import defaultdict
from gcn.LogHandler import LogHandler

class _SNE(object):

    def __init__(self, graph, lr=.001, rho=[.001,.001], mode_size=1000, rep_size=128, batch_size=100
                    , negative_ratio=5, order=3, identity_labels=None, table_size=1e8):
        '''
        graph: {'src':src_graph, 'obj':obj_graph}
        '''
        if not graph or not isinstance(graph, dict):
            print 'Graph should be a dictionary set of graphs'
            return

        self.epsilon = 1e-7
        self.table_size = table_size
        self.sigmoid_table = {}
        self.sigmoid_table_size = 1000
        self.SIGMOID_BOUND = 6

        self._init_simgoid_table()

        self.logger = LogHandler('sne_plain')

        keys = graph.keys()

        self.g = graph
        self.look_up = {key:self.g[key].look_up_dict for key in keys}
        # print self.look_up
        self.identity_labels = list(identity_labels)

        self.node_size = {key:graph[key].G.number_of_nodes() for key in keys}
        self.mode_size = mode_size
        self.rep_size = rep_size
        
        self._init_params(keys, self.node_size, mode_size, rep_size)

        self.order = order
        self.lr = lr
        self.rho = rho
        self.cur_epoch = 0
        self.batch_size = batch_size
        self.negative_ratio = negative_ratio

        self._gen_sampling_table()

    def _init_params(self, keys, node_size, mode_size, rep_size):
        self.codebook = np.random.normal(0,1,(mode_size,rep_size))
        self.embeddings = dict()
        self.content_embeddings = dict()
        self.b_e = dict()
        # self.b_e = defaultdict(list)
        for key in keys:
            self.embeddings[key] = np.random.normal(0,1,(node_size[key],mode_size))
            self.content_embeddings[key] = np.random.normal(0,1,(node_size[key],rep_size))
            self.b_e[key] = np.random.normal(0,1,(node_size[key],1))
            # for i in range(2):
            #     self.b_e[key].append(np.random.normal(0,1,(node_size[key],1)))

    def _init_simgoid_table(self):
        for k in range(self.sigmoid_table_size):
            x = 2*self.SIGMOID_BOUND*k/self.sigmoid_table_size-self.SIGMOID_BOUND
            self.sigmoid_table[k] = 1./(1+np.exp(-x))

    def _fast_sigmoid(self, val):
        if val>self.SIGMOID_BOUND:
            return 1-self.epsilon
        elif val<-self.SIGMOID_BOUND:
            return self.epsilon
        k = int((val+self.SIGMOID_BOUND)*self.sigmoid_table_size/self.SIGMOID_BOUND/2)
        return self.sigmoid_table[k]
        # return 1./(1+np.exp(-val))

    def _binarize(self, vec):
        return np.where(vec>=0, np.ones(vec.shape), np.zeros(vec.shape))

    def _update_graph_by_order2(self, x, key, batch):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        pos_h, pos_t, pos_h_v, neg_t = batch[key]
        batch_size = len(pos_h)

        # order 2
        pos_u_x = x[pos_h,:]
        pos_v_c = self.content_embeddings[key][pos_t,:]
        neg_u_x = x[pos_h_v,:]
        neg_v_c = self.content_embeddings[key][neg_t,:]

        pos_u = np.dot(pos_u_x, self.codebook)
        neg_u = np.dot(neg_u_x, self.codebook)

        # pos_e = np.sum(pos_u*pos_v_c, axis=1)+np.sum(self.b_e[key][1][pos_t,:], axis=1) # pos_e.shape = batch_size
        # neg_e = np.sum(neg_u*neg_v_c, axis=2)+np.sum(self.b_e[key][1][neg_t,:], axis=2) # neg_e.shape = batch_size*negative_ratio
        pos_e = np.sum(pos_u*pos_v_c, axis=1)+np.sum(self.b_e[key][pos_t,:], axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v_c, axis=2)+np.sum(self.b_e[key][neg_t,:], axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        # delta calculation
        delta_c = np.zeros(self.content_embeddings[key].shape)
        delta_x = np.zeros(self.embeddings[key].shape)
        delta_codebook = np.zeros(self.codebook.shape)
        # delta_b_e = np.zeros(self.b_e[key][1].shape) # delta_b_e in order 2
        delta_b_e = np.zeros(self.b_e[key].shape) # delta_b_e in order 2
        # temporal delta
        delta_eh = np.zeros((self.node_size[key],self.rep_size))

        # in postive cases
        for i in range(len(pos_t)):
            u,v = pos_h[i],pos_t[i]
            delta_b_e[v] += sigmoid_pos_e[i]-1
            delta_c[v] += (sigmoid_pos_e[i]-1)*pos_u[i,:]
            delta_eh[u] += (sigmoid_pos_e[i]-1)*pos_v_c[i,:]
        # in negative cases
        neg_shape = neg_e.shape
        for i in range(neg_shape[0]):
            for j in range(neg_shape[1]):
                u,v = pos_h_v[i][j],neg_t[i][j]
                delta_b_e[v] += sigmoid_neg_e[i,j]
                delta_c[v] += sigmoid_neg_e[i,j]*neg_u[i,j,:]
                delta_eh[u] += sigmoid_neg_e[i,j]*neg_v_c[i,j,:]

        # delta x & delta codebook
        delta_codebook = np.dot(x.T,delta_eh)
        delta_x = np.dot(delta_eh,self.codebook.T)

        return delta_x/batch_size, delta_c/batch_size, delta_codebook/batch_size, delta_b_e/batch_size

    def _update_graph_by_order1(self, x, key, batch):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        pos_h, pos_t, pos_h_v, neg_t = batch[key]
        batch_size = len(pos_h)

        # order 1
        pos_u_x = x[pos_h,:]
        pos_v_x = x[pos_t,:]
        neg_u_x = x[pos_h_v,:]
        neg_v_x = x[neg_t,:]

        pos_u = np.dot(pos_u_x, self.codebook)
        pos_v = np.dot(pos_v_x, self.codebook)
        neg_u = np.dot(neg_u_x, self.codebook)
        neg_v = np.dot(neg_v_x, self.codebook)

        # pos_e = np.sum(pos_u*pos_v, axis=1)+np.sum(self.b_e[key][0][pos_t,:], axis=1) # pos_e.shape = batch_size
        # neg_e = np.sum(neg_u*neg_v, axis=2)+np.sum(self.b_e[key][0][neg_t,:], axis=2) # neg_e.shape = batch_size*negative_ratio
        pos_e = np.sum(pos_u*pos_v, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        # delta calculation
        delta_x = np.zeros(self.embeddings[key].shape)
        delta_codebook = np.zeros(self.codebook.shape)
        # delta_b_e = np.zeros(self.b_e[key][0].shape) # delta b_e in order 1
        # temporal delta
        delta_eh = np.zeros((self.node_size[key],self.rep_size))

        # in postive cases
        for i in range(len(pos_t)):
            u,v = pos_h[i],pos_t[i]
            # delta_b_e[v] += sigmoid_pos_e[i]-1
            delta_eh[v] += (sigmoid_pos_e[i]-1)*pos_u[i,:]
            delta_eh[u] += (sigmoid_pos_e[i]-1)*pos_v[i,:]
        # in negative cases
        neg_shape = neg_e.shape
        for i in range(neg_shape[0]):
            for j in range(neg_shape[1]):
                u,v = pos_h_v[i][j],neg_t[i][j]
                # delta_b_e[v] += sigmoid_neg_e[i,j]
                delta_eh[v] += sigmoid_neg_e[i,j]*neg_u[i,j,:]
                delta_eh[u] += sigmoid_neg_e[i,j]*neg_v[i,j,:]

        # delta x & delta codebook
        delta_codebook = np.dot(x.T,delta_eh)
        delta_x = np.dot(delta_eh,self.codebook.T)

        # return delta_x/batch_size, delta_codebook/batch_size, delta_b_e/batch_size
        return delta_x/batch_size, delta_codebook/batch_size

    def _update_graph_by_orders(self, x, key, batch):
        '''
        x = self._binarize(self.embeddings[key])
        '''
        delta_b_e = [{} for i in range(2)]
        # order 2
        # delta_x_2, delta_c, delta_codebook_2, delta_b_e[1] = self._update_graph_by_order2(x, key, batch)
        delta_x_2, delta_c, delta_codebook_2, delta_b_e = self._update_graph_by_order2(x, key, batch)

        # order 1
        # delta_x_1, delta_codebook_1, delta_b_e[0] = self._update_graph_by_order1(x, key, batch)
        delta_x_1, delta_codebook_1 = self._update_graph_by_order1(x, key, batch)

        return delta_x_1+delta_x_2, delta_c, delta_codebook_1+delta_codebook_2, delta_b_e

    def _update_graph_by_identity(self, x_set, identity_labels):
        '''
        x = [self._binarize(self.embeddings[key]) for key in keys]
        '''
        keys = self.g.keys()
        if len(keys)!=2:
            return

        key0, key1 = keys[0], keys[1]

        code = dict()
        code[key0] = np.dot(x_set[key0], self.codebook)
        code[key1] = np.dot(x_set[key1], self.codebook)

        delta_h = dict()
        delta_h[key0] = np.zeros(code[key0].shape)
        delta_h[key1] = np.zeros(code[key1].shape)
        for label_u, label_v in identity_labels:
            u,v = self.look_up[key0][label_u], self.look_up[key1][label_v]
            delta_h[key0][u] = code[key0][u]-code[key1][v]
            delta_h[key1][v] = code[key1][v]-code[key0][u]

        delta_x = dict()
        delta_codebook = np.zeros(self.codebook.shape)
        for key in keys:
            delta_x[key] = np.dot(delta_h[key],self.codebook.T)
            delta_codebook += np.dot(x_set[key].T, delta_h[key])

        return delta_x, delta_codebook

    def update_graph(self, x_set, batch, identity_labels):
        keys = self.g.keys()

        delta_x = dict()
        delta_codebook = {}
        delta_c = dict()
        delta_b_e = dict()
        # update by identity labels
        tmp_delta_x, tmp_delta_codebook = self._update_graph_by_identity(x_set, identity_labels)
        delta_codebook = self.rho[0]*tmp_delta_codebook
        for key in keys:
            delta_x[key] = self.rho[0]*tmp_delta_x[key]
        # update by orders
        for key in keys:
            tmp_delta_x, delta_c[key], tmp_delta_codebook, delta_b_e[key]\
                        = self._update_graph_by_orders(x_set[key], key, batch)
            if key in delta_x:
                delta_x[key] += tmp_delta_x
            else:
                delta_x[key] = tmp_delta_x
            if len(delta_codebook)>1:
                delta_codebook += tmp_delta_codebook
            else:
                delta_codebook = tmp_delta_codebook

        return delta_x, delta_c, delta_codebook, delta_b_e

    def get_one_graph_loss_by_order2(self, x, key, batch):
        pos_h, pos_t, pos_h_v, neg_t = batch[key]

        # order 2
        pos_u_x = x[pos_h,:]
        pos_v_c = self.content_embeddings[key][pos_t,:]
        neg_u_x = x[pos_h_v,:]
        neg_v_c = self.content_embeddings[key][neg_t,:]

        pos_u = np.dot(pos_u_x, self.codebook)
        neg_u = np.dot(neg_u_x, self.codebook)

        # pos_e_2 = np.sum(pos_u*pos_v_c, axis=1)+np.sum(self.b_e[key][1][pos_t,:], axis=1) # pos_e.shape = batch_size
        # neg_e_2 = np.sum(neg_u*neg_v_c, axis=2)+np.sum(self.b_e[key][1][neg_t,:], axis=2) # neg_e.shape = batch_size*negative_ratio
        pos_e = np.sum(pos_u*pos_v_c, axis=1)+np.sum(self.b_e[key][pos_t,:], axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v_c, axis=2)+np.sum(self.b_e[key][neg_t,:], axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        return -np.mean(np.log(sigmoid_pos_e)+np.sum(np.log(1-sigmoid_neg_e), axis=1))

    def get_one_graph_loss_by_order1(self, x, key, batch):
        pos_h, pos_t, pos_h_v, neg_t = batch[key]

        # order 2
        pos_u_x = x[pos_h,:]
        pos_v_x = x[pos_t,:]
        neg_u_x = x[pos_h_v,:]
        neg_v_x = x[neg_t,:]

        pos_u = np.dot(pos_u_x, self.codebook)
        pos_v = np.dot(pos_v_x, self.codebook)
        neg_u = np.dot(neg_u_x, self.codebook)
        neg_v = np.dot(neg_v_x, self.codebook)

        # pos_e_1 = np.sum(pos_u*pos_v, axis=1)+np.sum(self.b_e[key][0][pos_t,:], axis=1) # pos_e.shape = batch_size
        # neg_e_1 = np.sum(neg_u*neg_v, axis=2)+np.sum(self.b_e[key][0][neg_t,:], axis=2) # neg_e.shape = batch_size*negative_ratio
        pos_e = np.sum(pos_u*pos_v, axis=1) # pos_e.shape = batch_size
        neg_e = np.sum(neg_u*neg_v, axis=2) # neg_e.shape = batch_size*negative_ratio

        sigmoid_pos_e = np.array([self._fast_sigmoid(val) for val in pos_e.reshape(-1)]).reshape(pos_e.shape)
        sigmoid_neg_e = np.array([self._fast_sigmoid(val) for val in neg_e.reshape(-1)]).reshape(neg_e.shape)

        return -np.mean(np.log(sigmoid_pos_e)+np.sum(np.log(1-sigmoid_neg_e), axis=1))

    def get_one_graph_loss(self, x, key, batch):
        return self.get_one_graph_loss_by_order1(x,key,batch)+self.get_one_graph_loss_by_order2(x,key,batch)

    def get_identity_loss(self, x_set, identity_labels):

        key0, key1 = 'src', 'obj'

        code = dict()
        code[key0] = np.dot(x_set[key0], self.codebook)
        code[key1] = np.dot(x_set[key1], self.codebook)

        u_set = tuple()
        v_set = tuple()
        for label_u, label_v in identity_labels:
            u_set += self.look_up[key0][label_u],
            v_set += self.look_up[key1][label_v],
        # print 'identity loss:',code[key0][u_set,:]-code[key1][v_set,:]
        return np.mean(np.linalg.norm(code[key0][u_set,:]-code[key1][v_set,:],axis=1))

    def train_one_epoch(self):
        sum_loss = 0.0
        batches = self.batch_iter()
        batch_id = 0
        for batch in batches:
            x_set = {key:self._binarize(self.embeddings[key]) for key in self.g.keys()}
            # print list(self.identity_labels)
            delta_x, delta_c, delta_codebook, delta_b_e = self.update_graph(x_set, batch, self.identity_labels)
            self.codebook -= self.lr*delta_codebook
            graph_loss = 0
            for key in self.g.keys():
                self.embeddings[key] -= np.clip(self.lr*delta_x[key],-1,1)
                self.content_embeddings[key] -= self.lr*delta_c[key]
                self.b_e[key] -= self.lr*delta_b_e[key]
                graph_loss += self.get_one_graph_loss(x_set[key], key, batch)
                # print 'u,v vec in {}:'.format(key), np.dot(x_set[key], self.codebook)
                # print key,'delta_embeddings:',np.clip(self.lr*delta_x[key],-1,1)
            identity_loss = self.get_identity_loss(x_set, self.identity_labels)
            cur_loss = graph_loss+self.rho[0]*identity_loss
            sum_loss += cur_loss
            batch_id += 1
            self.logger.info('Finish processing batch {} and cur_loss={}, identity_loss={}'
                                .format(batch_id, cur_loss, identity_loss))
            # print 'embedding:',self.embeddings
            # print 'detal_codebook',delta_codebook
            # print 'code_book',self.codebook
            # if batch_id%2==0:
            #     print 'Saving embedding in batch #{}'.format(batch_id)
            #     self.save_embeddings('tmp_embedding_{}'.format(batch_id))
        self.logger.info('Epoch:{}, sum of loss:{!s}'.format(self.cur_epoch, sum_loss/batch_id))
        self.cur_epoch += 1

    def cos_distance(vec1, vec2):
        # cosine similarity: vec1*vec2/(||vec1||*||vec2||)
        fac = np.dot(vec1, vec2)
        dem = np.linalg.norm(vec1)*np.linalg.norm(vec2)
        return fac/dem

    def geo_distance(vec1, vec2):
        # geo similarity: ||vec1-vec2||_2
        if len(vec1)!=len(vec2):
            print 'The lengths of vecs in geo_distance must be equal'
            return
        return np.linalg.norm(vec1-vec2)

    def get_random_node_pairs(self, i, key, shuffle_indices, edges, edge_set, numNodes):
        # balance the appearance of edges according to edge_prob
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
        return cur_h, cur_t, cur_h_v, cur_neg_t

    def batch_iter(self):

        numNodes = self.node_size

        keys = self.g.keys()

        edges = {key:[(self.look_up[key][x[0]], self.look_up[key][x[1]]) for x in self.g[key].G.edges()] for key in keys}
        data_size = {key:self.g[key].G.number_of_edges() for key in keys}
        edge_set = {key:set([x[0]*numNodes[key]+x[1] for x in edges[key]]) for key in keys}
        shuffle_indices = {key:np.random.permutation(np.arange(data_size[key])) for key in keys}

        start_index = 0
        min_data_size = min(data_size['src'], data_size['obj'])
        end_index = min(start_index+self.batch_size, min_data_size)
        while start_index < min_data_size:
            ret = dict()
            for key in keys:
                pos_h = []
                pos_t = []
                pos_h_v = []
                neg_t = []
                for i in range(start_index, end_index):
                    cur_h, cur_t, cur_h_v, cur_neg_t = self.get_random_node_pairs(i, key
                                                        , shuffle_indices, edges, edge_set, numNodes)
                    pos_h.append(cur_h)
                    pos_t.append(cur_t)
                    pos_h_v.append(cur_h_v)
                    neg_t.append(cur_neg_t)
                ret[key] = (pos_h, pos_t, pos_h_v, neg_t)

            start_index = end_index
            end_index = min(start_index+self.batch_size, min_data_size)

            yield ret

    def _gen_sampling_table(self):
        power = 0.75
        numNodes = self.node_size

        self.logger.info("Pre-procesing for non-uniform negative sampling!")
        node_degree = {key:np.zeros(numNodes[key]) for key in self.g.keys()} # out degree

        self.look_up = {key:self.g[key].look_up_dict for key in self.g.keys()}
        for key in self.g.keys():
            for edge in self.g[key].G.edges():
                node_degree[key][self.look_up[key][edge[0]]] += self.g[key].G[edge[0]][edge[1]]["weight"]

        norm = {key:sum([math.pow(node_degree[key][i], power) for i in range(numNodes[key])])\
                         for key in self.g.keys()}

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

    def get_one_embeddings(self, key, embeddings):
        vectors = dict()
        look_back = self.g[key].look_back_list
        for i, embedding in enumerate(embeddings):
            vectors[look_back[i]] = embedding
        return vectors

    def get_bias_e(self, key, bias_e):
        vectors = dict()
        look_back = self.g[key].look_back_list
        for i, b_e in enumerate(bias_e):
            vectors[look_back[i]] = b_e
        return vectors

    def get_codebook(self):
        return self.codebook

    def get_vectors(self):
        ret = dict()
        keys = self.g.keys()
        node_embeddings = dict()
        content_embeddings = dict()
        b_e = dict()
        for key in keys:
            node_embeddings[key]=self.get_one_embeddings(key, self.embeddings[key])
            content_embeddings[key]=self.get_one_embeddings(key, self.content_embeddings[key])
            b_e[key] = self.get_bias_e(key, self.b_e[key])

        ret['node_embeddings']=node_embeddings
        ret['content_embeddings']=content_embeddings
        ret['b_e']=b_e
        ret['codebook']=self.codebook
        return ret

class SNE(object):

    def __init__(self, graph, lr=.001, rho=[.001,.001], mode_size=1000, rep_size=128, batch_size=1000, epoch=10, 
                    negative_ratio=5, order=3, label_file=None, table_size=1e8, auto_stop=True):
        if not graph or not isinstance(graph, dict):
            print 'Parameter graph should be a dictionary set of graphs'
            return
        # paramter initialization
        self.g = graph
        self.mode_size = mode_size
        self.rep_size = rep_size
        self.order = order
        self.best_result = 0
        self.vectors = {}
        # training
        if label_file:
            identity_labels = self.read_identity_labels(filename=label_file)
            self.model = _SNE(graph, lr=lr, rho=rho, mode_size=mode_size, rep_size=rep_size
                                , batch_size=batch_size, negative_ratio=negative_ratio
                                , order=self.order, identity_labels=identity_labels
                                , table_size=table_size)
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
        self.vectors = self.model.get_vectors()

    def save_embeddings(self, filename):
        for c in self.vectors.keys():
            if 'node_embeddings' in c or 'content_embeddings' in c:
                for key in self.vectors[c].keys():
                    # filename-[node_embeddings/content-embeddings]-[src/obj]
                    fout = open('{}-{}-{}'.format(filename,c,key), 'w') 
                    node_num = len(self.vectors[c][key].keys())
                    fout.write("{} {}\n".format(node_num, self.mode_size))
                    for node, vec in self.vectors[c][key].items():
                        fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
                    fout.close()
            if 'b_e' in c:
                for key in self.vectors[c].keys():
                    # filename-[b_e]-[src/obj]
                    fout = open('{}-{}-{}'.format(filename,c,key), 'w') 
                    print self.vectors[c][key]
                    for i in range(2):
                        node_num = len(self.vectors[c][key].keys())
                        fout.write("{} order={}\n".format(node_num, i))
                        for node, vec in self.vectors[c][key].items():
                            fout.write("{} {}\n".format(node,' '.join([str(x) for x in vec])))
                    fout.close()
            if 'codebook' in c:
                # filename-[codebook]-[src/obj]
                fout = open('{}-{}'.format(filename,c), 'w') 
                fout.write("{} {}\n".format(self.mode_size, self.rep_size))
                for vec in self.vectors[c]:
                    fout.write("{}\n".format(' '.join([str(x) for x in vec])))
                fout.close()