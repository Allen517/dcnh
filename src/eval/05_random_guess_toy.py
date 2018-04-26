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

def eval_model(src_lbs, target_lbs, cand_size, model_file, out_file):

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
            pred_Y = np.random.rand(cand_size+1)
            pred_pos = (1+np.argwhere(np.argsort(pred_Y)[::-1]==0)[0][0])
            cur_mrr = 1./pred_pos
            mrr += cur_mrr
            cnt += 1
            # if cnt%100==0:
            #   print 'finish {}'.format(cnt)
            #   break
        mrr = mrr/cnt
        print 'processing exp, mrr={}'.format(mrr)
        fout.write('mean_mrr:{}, var:{}\n'.format(mrr, .0))

s = 0.3
c = 0.8
train_prob = 0.6

for k in range(5):
    for s in np.arange(.2, 1., .1):
        c = 0.8

        src_lbs, target_lbs = _read_labels('src/data_proc/toy_dat/labels/blogcatalog_labels.{}.test'.format(train_prob))

        eval_model(src_lbs, target_lbs, 9, 'blog.alp_model.random_guess.times_{}.train_{}.s_{}.c_{}.epoch'.format(k, train_prob, s, c), 'mrr.random_guess.toy')

    for c in np.arange(.1, 1., .1):
        s = 0.5
        if c==0.8:
            continue

        src_lbs, target_lbs = _read_labels('src/data_proc/toy_dat/labels/blogcatalog_labels.{}.test'.format(train_prob))

        eval_model(src_lbs, target_lbs, 9, 'blog.alp_model.random_guess.times_{}.train_{}.s_{}.c_{}.epoch'.format(k, train_prob, s, c), 'mrr.random_guess.toy')