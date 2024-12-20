import sys
import os
import argparse

parser = argparse.ArgumentParser(
    description="""Command-line bin creator.
Will read the entire content of the FASTA file into memory - beware.""",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    add_help=False)

parser.add_argument('fastapath', help='Path to FASTA file')
parser.add_argument('clusterspath', help='Path to clusters.tsv')
parser.add_argument('minsize', help='Minimum size of bin', type=int, default=0)
parser.add_argument('outdir', help='Directory to create')

if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()

args = parser.parse_args()

sys.path.append('../transBin')
import transBin

with open(args.clusterspath) as file:
    clusters = transBin.transBintools.read_clusters(file)

with transBin.transBintools.Reader(args.fastapath, 'rb') as file:
    fastadict = transBin.transBintools.loadfasta(file)

transBin.transBintools.write_bins(args.outdir, clusters, fastadict, maxbins=None, minsize=args.minsize)
