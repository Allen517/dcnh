import numpy as np
import random
from collections import defaultdict
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from lib_dcnh.dcnh_neg_share_params import *
from lib_dcnh.dcnh_neg_dep_params import *
from lib_dcnh.dcn_neg_share_params import *
from lib_dcnh.pale_mlp import *
from lib_dcnh.pale_lin import *
from lib_dcnh.MNA import *
from lib_dcnh.graph import *
import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--embedding1', required=False,
                        help='Input embeddings of douban')
    parser.add_argument('--embedding2', required=False,
                        help='Input embeddings of weibo')
    parser.add_argument('--graph1', required=False,
                        help='Network of douban')
    parser.add_argument('--graph2', required=False,
                        help='Network of weibo')
    parser.add_argument('--identity-linkage', required=True,
                        help='Input linkage of douban2weibo')
    parser.add_argument('--output', required=True,
                        help='Output model file')
    parser.add_argument('--log-file', default='DCNH',
                        help='logging file')
    parser.add_argument('--lr', default=.01, type=float,
                        help='Learning rate')
    parser.add_argument('--batch-size', default=128, type=int,
                        help='Batch size')
    parser.add_argument('--input-size', default=256, type=int,
                        help='Number of embedding')
    parser.add_argument('--hidden-size', default=32, type=int,
                        help='Number of embedding')
    parser.add_argument('--output-size', default=32, type=int,
                        help='Number of output code')
    parser.add_argument('--layers', default=2, type=int,
                        help='Number of layers')
    parser.add_argument('--saving-step', default=1, type=int,
                        help='The training epochs')
    parser.add_argument('--max-epochs', default=21, type=int,
                        help='The training epochs')
    parser.add_argument('--method', required=True, choices=['dcnh-sp', 'dcnh-dp', 'dcn-sp', 'pale-mlp', 'pale-lin', 'mna'],
                        help='The learning methods')
    parser.add_argument('--neg-ratio', default=5, type=int,
                        help='The negative ratio of LINE')
    parser.add_argument('--device', default=':/gpu:0',
                        help='Running device')
    args = parser.parse_args()
    return args

def main(args):
    t1 = time.time()
    SAVING_STEP = args.saving_step
    MAX_EPOCHS = args.max_epochs
    if args.method == 'dcnh-sp':
        model = DCNH_SP(learning_rate=args.lr, batch_size=args.batch_size
                        , neg_ratio=args.neg_ratio, n_input=args.input_size
                        , n_out=args.output_size, n_hidden=args.hidden_size, n_layer=args.layers
                        , files=[args.embedding1, args.embedding2, args.identity_linkage]
                        , log_file=args.log_file, device=args.device)
    if args.method == 'dcnh-dp':
        model = DCNH_DP(learning_rate=args.lr, batch_size=args.batch_size
                        , neg_ratio=args.neg_ratio, n_input=args.input_size
                        , n_out=args.output_size, n_hidden=args.hidden_size, n_layer=args.layers
                        , files=[args.embedding1, args.embedding2, args.identity_linkage]
                        , log_file=args.log_file, device=args.device)
    if args.method == 'dcn-sp':
        model = DCN_SP(learning_rate=args.lr, batch_size=args.batch_size
                        , neg_ratio=args.neg_ratio, n_input=args.input_size
                        , n_out=args.output_size, n_hidden=args.hidden_size, n_layer=args.layers
                        , files=[args.embedding1, args.embedding2, args.identity_linkage]
                        , log_file=args.log_file, device=args.device)
    if args.method == 'pale-mlp':
        model = PALE_MLP(learning_rate=args.lr, batch_size=args.batch_size
                        , n_input=args.input_size, n_hidden=args.hidden_size, n_layer=args.layers
                        , files=[args.embedding1, args.embedding2, args.identity_linkage]
                        , log_file=args.log_file, device=args.device)
    if args.method == 'pale-lin':
        model = PALE_LIN(learning_rate=args.lr, batch_size=args.batch_size
                        , n_input=args.input_size, files=[args.embedding1, args.embedding2, args.identity_linkage]
                        , log_file=args.log_file, device=args.device)
    if args.method == 'mna':
        graph = defaultdict(Graph)
        print "Loading graph..."
        if args.graph1:
            graph['f'].read_adjlist(filename=args.graph1)
        if args.graph2:
            graph['g'].read_adjlist(filename=args.graph2)
        model = MNA(graph=graph, anchorfile=args.identity_linkage, valid_prop=1., neg_ratio=3, log_file=args.log_file)

    print args.embedding1, args.embedding2
    if args.method != 'mna':
        for i in range(1,MAX_EPOCHS+1):
            model.train_one_epoch()
            if i>0 and i%SAVING_STEP==0:
                model.save_models(args.output+'.epoch_'+str(i))
    else:
        model.save_model(args.output)
    t2 = time.time()
    print 'time cost:',t2-t1

if __name__ == "__main__":
    main(parse_args())
