# -*- coding:utf8 -*-

import numpy as np

def sample_case(s_prob):
	k = 0
	while True:
		if k<len(s_prob):
			rd = np.random.rand()
			if rd<s_prob[k]:
				return k
		else:
			return len(s_prob)
		k+=1

def sample_prob(alpha_s, alpha_c):
	sparsity = 1-alpha_s
	overlap = alpha_c
	return 1-2*sparsity+sparsity*overlap, 1-sparsity, 1-sparsity*overlap

src_netfile = 'data/blogCatalog/bc_adjlist.txt'

net1_file = 'data/blogCatalog/net1'
net2_file = 'data/blogCatalog/net2'

alpha_s = 0.2
alpha_c = 0.8

s_prob = sample_prob(alpha_s, alpha_c)

print s_prob

net1_f_handler = open(net1_file, 'w')
net2_f_handler = open(net2_file, 'w')
with open(src_netfile, 'r') as f_handler:
	for ln in f_handler:
		ln = ln.strip()
		if ln:
			elems = ln.split()
		nd = elems[0]
		net1_links = ''
		net2_links = ''
		cnt = 0
		for i in range(1, len(elems)):
			prob_index = sample_case(s_prob)
			if prob_index==1:
				net1_links += elems[i]+' '
			if prob_index==2:
				net2_links += elems[i]+' '
			if prob_index==3:
				cnt += 1
				net1_links += elems[i]+' '
				net2_links += elems[i]+' '
		net1_f_handler.write(nd+' '+net1_links+'\n')
		net2_f_handler.write(nd+' '+net2_links+'\n')


