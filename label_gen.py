
import numpy as np

filepath = 'data/blogCatalog/bc_adjlist.txt'
outFilePath = 'data/blogCatalog/align_labels_p_0dot1'

nodeSet = set()

with open(filepath, 'r') as objF:
	for ln in objF:
		nds = ln.strip().split()
		nodeSet.update(nds)

with open(outFilePath, 'w') as objF:
	for nd in nodeSet:
		if np.random.rand()<0.1:
			objF.write(nd+' '+nd+'\n')
