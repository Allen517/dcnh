# -*- coding=UTF-8 -*-\n
import numpy as np
import random
from collections import defaultdict
import json
import re,os

def get_test_loss(filename):
    if not os.path.exists(filename):
        return None
    test_loss = tuple()
    with open(filename, 'r') as fin:
        for ln in fin:
            ln = ln.strip()
            if 'Epoch' in ln:
                p = re.compile(r'sum of loss=(-\d+.\d+|\d+.\d+), mrr=(-\d+.\d+|\d+.\d+)')
                match = p.search(ln)
                if match:
                    test_loss += match.group(2),
    return test_loss

def get_model_perf(base_log_path, base_res_path, base_model_path, base_label_path, log_rcd_file, out_file, range_from, range_to, gap):
    skip_step = 100
    cand_size = 9
    with open(log_rcd_file, 'aw') as fout:
        method = 'dcnh-sp'
        j = 0.9
        for k in np.arange(range_from, range_to, gap):
            src_lbs, target_lbs = _read_labels('{}.{}.test'.format(base_label_path, j))
            filename='{}/{}.d2w.case.times_{}.log'.format(base_log_path, method, k)
            print filename
            test_loss = get_test_loss(filename)
            if test_loss:
                test_loss = map(float, test_loss[::skip_step])
                max_idx = np.argmax(test_loss)
                fout.write('{}: max test loss:{} @{}\n'.format(filename, max(test_loss), np.argmax(test_loss)))
                if method=='dcnh-sp':
                    eval_hashing_model(src_lbs, target_lbs, 'd2w_embedding/douban_all.txt', 'd2w_embedding/weibo_all.txt', '{}/d2w.alp_model.case.{}.times_{}.epoch_{}'.format(base_model_path, method, k, max_idx*skip_step), cand_size, out_file, calc_mlp_sp_res, _hamming_distance)


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

def _read_embeddings(filename):
    if not os.path.exists(filename):
        return None
    print 'reading {}'.format(filename)
    embedding = dict()
    with open(filename, 'r') as f_handler:
        for ln in f_handler:
            ln = ln.strip()
            if ln:
                elems = ln.split()
                if len(elems)==2:
                    continue
                embedding[elems[0]] = map(float, elems[1:])
    return embedding

def _read_model(filename):
    print 'reading model {}'.format(filename)
    if not os.path.exists(filename):
        return None
    model = defaultdict(list)
    with open(filename, 'r') as f_handler:
        cur_key = ''
        for ln in f_handler:
            ln = ln.strip()
            if 'h' in ln or 'b' in ln or 'out' in ln:
                cur_key = ln
                continue
            model[cur_key].append(map(float, ln.split()))
    return model

def _tanh(mat):
    # (1-exp(-2x))/(1+exp(-2x))
    mat = np.array(mat)
    return np.tanh(mat)

def _sigmoid(mat):
    # 1/(1+exp(-mat))
    mat = np.array(mat)
    return 1./(1+np.exp(-mat))

def _hamming_distance(vec1, vec2):
    res = np.where(vec1*vec2<0, np.ones(vec1.shape), np.zeros(vec1.shape))
    return np.sum(res)

def _dot_distance(vec1, vec2):
    return -np.sum(vec1*vec2)

def _geo_distance(vec1, vec2):
    return .5*np.sum((vec1-vec2)**2)

def calc_mlp_sp_res(inputs, n_layer, model, model_type):
    inputs = np.array(inputs)

    layer = _sigmoid(np.dot(inputs,np.array(model['h0_'+model_type]))+np.array(model['b0_'+model_type]).reshape(-1))
    for i in range(1, n_layer):
        layer = _sigmoid(np.dot(layer,np.array(model['h{}'.format(i)]))+np.array(model['b{}'.format(i)]).reshape(-1))
    out = _tanh(np.dot(layer,np.array(model['out']))+np.array(model['b_out']).reshape(-1))

    return out

def calc_mlp_dp_res(inputs, n_layer, model, model_type):
    inputs = np.array(inputs)

    layer = _sigmoid(np.dot(inputs,np.array(model['h0_'+model_type]))+np.array(model['b0_'+model_type]).reshape(-1))
    for i in range(1, n_layer):
        layer = _sigmoid(np.dot(layer,np.array(model['h{}_{}'.format(i,model_type)]))+np.array(model['b{}_{}'.format(i,model_type)]).reshape(-1))
    out = _tanh(np.dot(layer,np.array(model['out_'+model_type]))+np.array(model['b_out_'+model_type]).reshape(-1))

    return out

def calc_mlp_res(inputs, n_layer, model):
    inputs = np.array(inputs)

    layer = _sigmoid(np.dot(inputs,np.array(model['h0']))+np.array(model['b0']).reshape(-1))
    for i in range(1, n_layer):
        layer = _sigmoid(np.dot(layer,np.array(model['h{}'.format(i)]))+np.array(model['b{}'.format(i)]).reshape(-1))
    out = _tanh(np.dot(layer,np.array(model['out']))+np.array(model['b_out']).reshape(-1))

    return out

def calc_lin_res(inputs, n_layer, model):
    inputs = np.array(inputs)

    out = _tanh(np.dot(inputs,np.array(model['out']))+np.array(model['b_out']).reshape(-1))

    return out

def eval_hashing_model(src_lbs, target_lbs, src_emb_file, target_emb_file, model_file, cand_size, out_file, calc_model_res, dist_calc):
    print 'processing {} and {}'.format(src_emb_file, target_emb_file)
    if not os.path.exists(src_emb_file) or not os.path.exists(target_emb_file):
        print 'file not found...'
        return

    src_embedding = _read_embeddings(src_emb_file)
    target_embedding = _read_embeddings(target_emb_file)

    model = _read_model(model_file)
    if not model:
        return 

    mrr_list = list()
    target_keys = target_embedding.keys()
    if len(src_lbs)!=len(target_lbs):
        print 'the label size in src_lbs and target_lbs is not equal!'
        return
    lb_size = len(src_lbs)

    with open(out_file, 'aw') as fout:
        fout.write(model_file+"\n")
        mrr_list = tuple()
        for m in range(1):
            cnt = 0
            mrr = .0
            for i in range(lb_size):
                dou_id = src_lbs[i]
                w_id = target_lbs[i]

                if not dou_id in src_embedding or not w_id in target_embedding:
                    continue

                src_e = src_embedding[dou_id]
                target_e = target_embedding[w_id]

                model_res_f = calc_model_res(src_e, 1, model, 'f')
                model_res_g = calc_model_res(target_e, 1, model, 'g')

                anchor_dist = dist_calc(model_res_f, model_res_g)

                pred_pos = 1
                noise_ids = set()
                for k in range(cand_size):
                    rand_lb_id = np.random.randint(0, lb_size)
                    noise_w_id = target_lbs[rand_lb_id]
                    while noise_w_id==w_id or noise_w_id in noise_ids or not noise_w_id in target_embedding:
                        rand_lb_id = np.random.randint(0, lb_size)
                        noise_w_id = target_lbs[rand_lb_id]
                    noise_dist = dist_calc(model_res_f, 
                                        calc_model_res(target_embedding[noise_w_id], 1, model, 'g'))
                    if noise_dist<anchor_dist:
                        pred_pos += 1
                cur_mrr = 1./pred_pos
                mrr += cur_mrr
                cnt += 1
            mrr = mrr/cnt
            print 'processing {}-th exp, mrr={}'.format(m, mrr)
            mrr_list += mrr,
        fout.write('mean_mrr:{}, var:{}\n'.format(np.mean(mrr_list), np.var(mrr_list)))

def eval_pale_model(src_lbs, target_lbs, src_emb_file, target_emb_file, model_file, cand_size, out_file, calc_model_res, dist_calc):
    print 'processing {} and {}'.format(src_emb_file, target_emb_file)
    if not os.path.exists(src_emb_file) or not os.path.exists(target_emb_file):
        print 'file not found...'
        return

    src_embedding = _read_embeddings(src_emb_file)
    target_embedding = _read_embeddings(target_emb_file)

    model = _read_model(model_file)
    if not model:
        return 

    mrr_list = list()
    target_keys = target_embedding.keys()
    if len(src_lbs)!=len(target_lbs):
        print 'the label size in src_lbs and target_lbs is not equal!'
        return
    lb_size = len(src_lbs)

    with open(out_file, 'aw') as fout:
        fout.write(model_file+"\n")
        mrr_list = list()
        for m in range(1):
            cnt = 0
            mrr = .0
            for i in range(lb_size):
                dou_id = src_lbs[i]
                w_id = target_lbs[i]

                if not dou_id in src_embedding or not w_id in target_embedding:
                    continue

                src_e = src_embedding[dou_id]
                target_e = target_embedding[w_id]

                model_res_f = calc_model_res(src_e, 1, model)
                model_res_g = np.array(target_e)

                anchor_dist = dist_calc(model_res_f, model_res_g)

                pred_pos = 1
                noise_ids = set()
                for k in range(cand_size):
                    rand_lb_id = np.random.randint(0, lb_size)
                    noise_w_id = target_lbs[rand_lb_id]
                    while noise_w_id==w_id or noise_w_id in noise_ids or not noise_w_id in target_embedding:
                        rand_lb_id = np.random.randint(0, lb_size)
                        noise_w_id = target_lbs[rand_lb_id]
                    noise_dist = dist_calc(model_res_f, np.array(target_embedding[noise_w_id]))
                    if noise_dist<anchor_dist:
                        pred_pos += 1
                cur_mrr = 1./pred_pos
                mrr += cur_mrr
                cnt += 1
            mrr = mrr/cnt
            print 'processing {}-th exp, mrr={}'.format(m, mrr)
            mrr_list += mrr,
        fout.write('mean_mrr:{}, var:{}\n'.format(np.mean(mrr_list), np.var(mrr_list)))


get_model_perf('log', 'res', 'anchor_res', 'data/douban2weibo/d2w.anchor_links.labels', 'log_analy.case.d2w', 'mrr.case.d2w', 1, 6, 1)

