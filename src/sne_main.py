import numpy as np
import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from sklearn.linear_model import LogisticRegression
from libnrl.graph import *
from libnrl.sne import SNE
import time

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input1', required=True,
                        help='Input first graph file')
    parser.add_argument('--input2', required=True,
                        help='Input second graph file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--directed', action='store_true',
                        help='Treat graph as directed.')
    parser.add_argument('--lr', default=.001, type=float,
                        help='Learning rate')
    parser.add_argument('--batch-size', default=1000, type=int,
                        help='Batch size')
    parser.add_argument('--representation-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--method', required=True, choices=['sne'],
                        help='The learning method')
    parser.add_argument('--label-file', default='',
                        help='The file of node label')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--negative-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--weighted', action='store_true',
                        help='Treat graph as weighted')
    parser.add_argument('--clf-ratio', default=0.5, type=float,
                        help='The ratio of training data in the classification')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--auto-stop', action='store_true',
                        help='Using early stop when training LINE')
    args = parser.parse_args()
    return args


def main(args):
    t1 = time.time()
    g_src = Graph()
    g_obj = Graph()
    print "Reading..."
    if args.graph_format == 'adjlist':
        g_src.read_adjlist(filename=args.input1)
        g_obj.read_adjlist(filename=args.input2)
    elif args.graph_format == 'edgelist':
        g_src.read_edgelist(filename=args.input1, weighted=args.weighted, directed=args.directed)
        g_obj.read_edgelist(filename=args.input2, weighted=args.weighted, directed=args.directed)
    g = {'src':g_src, 'obj':g_obj}
    if args.method == 'sne':
        if args.label_file:
            model = SNE(g, lr=args.lr, batch_size=args.batch_size, epoch=args.epochs,
                                rep_size=args.representation_size, order=args.order, 
                                label_file=args.label_file, clf_ratio=args.clf_ratio, 
                                auto_stop=args.auto_stop)
        else:
            model = SNE(g, lr=args.lr, batch_size=args.batch_size, epoch=args.epochs, 
                                rep_size=args.representation_size, order=args.order)
    t2 = time.time()
    print t2-t1
    if args.method == 'sne':
        print "Saving embeddings..."
        model.save_embeddings(args.output)

if __name__ == "__main__":
    random.seed(32)
    np.random.seed(32)
    main(parse_args())
