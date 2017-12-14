
import numpy as np

# filepath = 'data/blogCatalog/bc_adjlist.txt'
# outFilePath = 'data/blogCatalog/align_labels_p_0dot1'

# nodeSet = set()

# with open(filepath, 'r') as objF:
# 	for ln in objF:
# 		nds = ln.strip().split()
# 		nodeSet.update(nds)

# with open(outFilePath, 'w') as objF:
# 	for nd in nodeSet:
# 		if np.random.rand()<0.1:
# 			objF.write(nd+' '+nd+'\n')


filepath = '/home/yqwang/Codes/python/sparse_network_embedding/src/data_proc/douban2weibo/douban_weibo.identity.users.final'
outFilePath = '/home/yqwang/Codes/python/sparse_network_embedding/src/data_proc/douban2weibo/douban_weibo.identity.users.final.p0dot8'

p=0.8

with open(filepath, 'r') as in_handler, open(outFilePath, 'w') as out_handler:
	for ln in in_handler:
		wrtLn = ''
		nds = ln.strip().split(',')
		if len(nds)<2:
			continue
		d_nd = nds[0]
		w_nds = nds[1].split(';')
		if len(w_nds)>1:
			for uid in w_nds:
				if np.random.rand()<p:
					wrtLn += d_nd+' '+uid+'\n'
		else:
			if np.random.rand()<p:
				wrtLn += d_nd+' '+w_nds[0]+'\n'
		out_handler.write(wrtLn)
