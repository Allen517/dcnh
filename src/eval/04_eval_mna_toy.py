import sys
import os
sys.path.append('../')

import numpy as np
from sklearn.externals import joblib
from sklearn import svm
from collections import defaultdict

from graph import *

def _read_model(modelfile):
    return joblib.load(modelfile)

def _read_labels(filename):
    if not os.path.exists(filename):
        return None, None
    # read labels
    src_lbs = tuple()
    target_lbs = tuple()

    with open(filename, 'r') as lb_f_handler:
        for ln in lb_f_handler:
            ln = ln.strip()
            if ln:
                labels = ln.split()
                dou_id = labels[0]
                w_id = labels[1]
            src_lbs += dou_id,
            target_lbs += w_id,
    return src_lbs, target_lbs

def _get_pair_features(src_nds, target_nds, src_lbs, target_lbs, graph):
    pair_features = list()
    if len(src_nds)!=len(target_nds):
        print 'The size of sampling in processing __get_pair_features is not equal.'
        yield pair_features
    for i in range(len(src_nds)):
        src_nd, target_nd = src_nds[i],target_nds[i]

        if not src_nd in graph['f'].G or not target_nd in graph['g'].G:
            continue

        src_neighbor_anchors = set()
        for src_nd_to in graph['f'].G[src_nd]:
            if src_nd_to in src_lbs:
                src_neighbor_anchors.add(src_nd_to)

        target_neighbor_anchors = set()
        for target_nd_to in graph['g'].G[target_nd]:
            if target_nd_to in target_lbs:
                target_neighbor_anchors.add(target_nd_to)

        cnt_common_neighbors = .0
        AA_measure = .0
        for sna in src_neighbor_anchors:
            target_anchor_nd = target_lbs[src_lbs.index(sna)]
            if target_anchor_nd in target_neighbor_anchors:
                cnt_common_neighbors += 1.
                AA_measure += 1./np.log((len(graph['f'].G[sna])+len(graph['g'].G[target_anchor_nd]))/2.)
        jaccard = cnt_common_neighbors/(len(graph['f'].G[src_nd])\
                                        +len(graph['g'].G[target_nd])-cnt_common_neighbors+1e-6)

        yield [cnt_common_neighbors, jaccard, AA_measure]

def eval_model(graph, src_lbs, target_lbs, model_file, cand_size, out_file):

    print 'Read model {}'.format(model_file)
    model = _read_model(model_file)
    if not model:
        return 

    mrr_list = list()
    if len(src_lbs)!=len(target_lbs):
        print 'the label size in src_lbs and target_lbs is not equal!'
        return
    lb_size = len(src_lbs)

    with open(out_file, 'aw') as fout:
        fout.write(model_file+"\n")
        cnt = 0
        mrr = .0
        for i in range(lb_size):
            dou_id_list = list()
            w_id_list = list()
            dou_id = src_lbs[i]
            w_id = target_lbs[i]

            dou_id_list.append(dou_id)
            w_id_list.append(w_id_list)

            noise_ids = set()
            for k in range(cand_size):
                rand_lb_id = np.random.randint(0, lb_size)
                noise_w_id = target_lbs[rand_lb_id]
                while noise_w_id==w_id or noise_w_id in noise_ids:
                    rand_lb_id = np.random.randint(0, lb_size)
                    noise_w_id = target_lbs[rand_lb_id]
                dou_id_list.append(dou_id)
                w_id_list.append(noise_w_id)
            pred_X = list(_get_pair_features(dou_id_list, w_id_list, src_lbs, target_lbs, graph))
            pred_Y = model.predict(pred_X)
            pred_pos = (1+np.argwhere(np.argsort(pred_Y)[::-1]==0)[0][0])
            cur_mrr = 1./pred_pos
            mrr += cur_mrr
            cnt += 1
            if cnt%100==0:
            	print 'finish {}'.format(cnt)
            	break
        mrr = mrr/cnt
        print 'processing exp, mrr={}'.format(mrr)
        fout.write('mean_mrr:{}, var:{}\n'.format(mrr, .0))

s = 0.3
c = 0.8
train_prob = 0.6

for s in np.arange(.2, 1., .1):
    c = 0.8
    graph = defaultdict(Graph)
    print "Loading graph {} {}".format('src/data_proc/toy_dat/subnets/blogcatalog_net.src.s_{}.c_{}'.format(s, c),\
                                'src/data_proc/toy_dat/subnets/blogcatalog_net.target.s_{}.c_{}'.format(s, c))
    graph['f'].read_adjlist(filename='src/data_proc/toy_dat/subnets/blogcatalog_net.src.s_{}.c_{}'.format(s, c))
    graph['g'].read_adjlist(filename='src/data_proc/toy_dat/subnets/blogcatalog_net.target.s_{}.c_{}'.format(s, c))

    src_lbs, target_lbs = _read_labels('src/data_proc/toy_dat/labels/blogcatalog_labels.{}.test'.format(train_prob))

    eval_model(graph, src_lbs, target_lbs, 'anchor_res/blog.alp_model.mna.times_{}.train_{}.s_{}.c_{}.pkl'.format(1, train_prob, s, c),\
                9, 'mrr.mna.toy')

for c in np.arange(.1, 1., .1):
    s = 0.5
    if c==0.8:
        continue
    graph = defaultdict(Graph)
    print "Loading graph {} {}".format('src/data_proc/toy_dat/subnets/blogcatalog_net.src.s_{}.c_{}'.format(s, c),\
                                'src/data_proc/toy_dat/subnets/blogcatalog_net.target.s_{}.c_{}'.format(s, c))
    graph['f'].read_adjlist(filename='src/data_proc/toy_dat/subnets/blogcatalog_net.src.s_{}.c_{}'.format(s, c))
    graph['g'].read_adjlist(filename='src/data_proc/toy_dat/subnets/blogcatalog_net.target.s_{}.c_{}'.format(s, c))

    src_lbs, target_lbs = _read_labels('src/data_proc/toy_dat/labels/blogcatalog_labels.{}.test'.format(train_prob))

    eval_model(graph, src_lbs, target_lbs, 'anchor_res/blog.alp_model.mna.times_{}.train_{}.s_{}.c_{}.pkl'.format(1, train_prob, s, c),\
                9, 'mrr.mna.toy')