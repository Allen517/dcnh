import random
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from n2v.line_align_pretrain import LINE_ALIGN_PRETRAIN
from n2v.line_anchor import LINE_ANCHORREG_ALIGN_PRETRAIN

def parse_args():
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter,
                            conflict_handler='resolve')
    parser.add_argument('--input', required=True,
                        help='Input graph file')
    parser.add_argument('--output', required=True,
                        help='Output representation file')
    parser.add_argument('--last-emb-file', required=False,
                        help='Representation file from last training')
    parser.add_argument('--embed-file', required=False,
                        help='Representation file from src graph')
    parser.add_argument('--anchor-file', required=False,
                        help='Anchor links file')
    parser.add_argument('--log-file', default='log',
                        help='logging file')
    parser.add_argument('--lr', default=.001, type=float,
                        help='Learning rate')
    parser.add_argument('--gamma', default=.01, type=float,
                        help='Gamma')
    parser.add_argument('--batch-size', default=1000, type=int,
                        help='Batch size')
    parser.add_argument('--table-size', default=1e8, type=int,
                        help='Table size')
    parser.add_argument('--rep-size', default=128, type=int,
                        help='Number of latent dimensions to learn for each node.')
    parser.add_argument('--epochs', default=5, type=int,
                        help='The training epochs of LINE and GCN')
    parser.add_argument('--method', required=True, choices=['line_align', 'line', 'linex', 'line_anchor_reg'],
                        help='The learning method')
    parser.add_argument('--graph-format', default='adjlist', choices=['adjlist', 'edgelist'],
                        help='Input graph format')
    parser.add_argument('--neg-ratio', default=5, type=int,
                        help='the negative ratio of LINE')
    parser.add_argument('--order', default=3, type=int,
                        help='Choose the order of LINE, 1 means first order, 2 means second order, 3 means first order + second order')
    parser.add_argument('--no-auto-stop', action='store_true',
                        help='no early stop when training LINE')
    args = parser.parse_args()
    return args


def main(args):
    if 'x' in args.method:
        from n2v.utils.graphx import Graph
        from n2v.linex import LINE
    else:
        from n2v.utils.graph import Graph
        from n2v.line import LINE
    g = Graph()
    print "Reading..."
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted, directed=args.directed)
    if args.method == 'line' or args.method=='linex':
        model = LINE(g, lr=args.lr, batch_size=args.batch_size, epoch=args.epochs, 
                            rep_size=args.rep_size, table_size=args.table_size,
                            order=args.order, outfile=args.output, log_file=args.log_file, 
                            last_emb_file=args.last_emb_file, negative_ratio=args.neg_ratio)
    if args.method == 'line_align':
        model = LINE_ALIGN_PRETRAIN(g, lr=args.lr, batch_size=args.batch_size, epoch=args.epochs, 
                            rep_size=args.rep_size, table_size=args.table_size, negative_ratio=args.neg_ratio,
                            order=args.order, outfile=args.output, anchorfile=args.anchor_file,
                            embedfile=args.embed_file, log_file=args.log_file)
    if args.method == 'line_anchor_reg':
        model = LINE_ANCHORREG_ALIGN_PRETRAIN(g, lr=args.lr, gamma=args.gamma, batch_size=args.batch_size, epoch=args.epochs, 
                            rep_size=args.rep_size, table_size=args.table_size, negative_ratio=args.neg_ratio,
                            order=args.order, outfile=args.output, anchorfile=args.anchor_file,
                            embedfile=args.embed_file, log_file=args.log_file)

if __name__ == "__main__":
    main(parse_args())
