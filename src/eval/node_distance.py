# -*- coding:utf8 -*-

import numpy as np

def read_node_embedding(filename):
	look_up = dict()
	embedding = list()
	with open(filename, 'r') as f_handler:
		isFirstLn = True
		cnt = 0
		for ln in f_handler:
			if isFirstLn:
				isFirstLn = False
				continue
			elems = ln.strip().split()
			node = elems[0]
			vec = map(float, elems[1:])
			look_up[node] = cnt
			embedding.append(vec)
			cnt += 1
	return look_up, np.array(embedding)
	# return look_up, _binarize(np.array(embedding))

def read_codebook(filename):
	codebook = list()
	with open(filename, 'r') as f_handler:
		isFirstLn = True
		cnt = 0
		for ln in f_handler:
			if isFirstLn:
				isFirstLn = False
				continue
			elems = ln.strip().split()
			vec = map(float, elems)
			codebook.append(vec)
	return np.array(codebook)

def _binarize(vec):
        return np.where(vec>=0, np.ones(vec.shape), np.zeros(vec.shape))

def get_identity_labels(filename):
	with open(filename, 'r') as fin:
		for line in fin:
		    ln = line.strip()
		    if not ln:
		        break
		    yield ln.split()

def get_node_vec(iter):
	look_up = dict()
	codebook = read_codebook('epoch_res_{}.codebook'.format(iter))
	look_up['src'],embedding_src = read_node_embedding('epoch_res_{}.node_embeddings_src'.format(iter))
	look_up['obj'],embedding_obj = read_node_embedding('epoch_res_{}.node_embeddings_obj'.format(iter))
	return look_up,np.dot(embedding_src, codebook), np.dot(embedding_obj, codebook)

def geo_distance(u_vec, v_vec, lookup, identity_labels):
	u_set = tuple()
	v_set = tuple()
	for label_u, label_v in identity_labels:
		u_set += lookup['src'][label_u],
		v_set += lookup['obj'][label_v],
	print u_vec[u_set,:],u_set
	print v_vec[v_set,:],v_set
	return np.mean(np.linalg.norm(u_vec[u_set,:]-v_vec[v_set,:], axis=1))

def geo_distance_full(u_vec, v_vec, lookup):
	u_set = tuple()
	v_set = tuple()
	for key in lookup['src'].keys():
		u_set += lookup['src'][key],
		v_set += lookup['obj'][key],
	print u_vec[u_set,:],u_set
	print v_vec[v_set,:],v_set
	return np.mean(np.linalg.norm(u_vec[u_set,:]-v_vec[v_set,:], axis=1))

lookup0, u_vec0, v_vec0=get_node_vec(0)
lookup8, u_vec8, v_vec8=get_node_vec(998)

identity_labels = list(get_identity_labels('data/test.align'))

print geo_distance(u_vec0,v_vec0, lookup0, identity_labels)
print geo_distance(u_vec8,v_vec8, lookup8, identity_labels)

print geo_distance_full(u_vec0,v_vec0, lookup0)
print geo_distance_full(u_vec8,v_vec8, lookup8)
