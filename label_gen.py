
filepath = 'data/blogCatalog/bc_adjlist.txt'
outFilePath = 'data/blogCatalog/align_labels'

nodeSet = set()

with open(filepath, 'r') as objF:
	for ln in objF:
		nds = ln.strip().split()
		nodeSet.update(nds)

with open(outFilePath, 'w') as objF:
	for nd in nodeSet:
		objF.write(nd+' '+nd+'\n')
