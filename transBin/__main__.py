#!/usr/bin/env python3

# More imports below, but the user's choice of processors must be parsed before
# numpy can be imported.
import sys
import os
import argparse
import torch
import datetime
import time
import shutil
import optuna

DEFAULT_PROCESSES = min(os.cpu_count(), 8)

# These MUST be set before importing numpy
# I know this is a shitty hack, see https://github.com/numpy/numpy/issues/11826
os.environ["MKL_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["NUMEXPR_NUM_THREADS"] = str(DEFAULT_PROCESSES)
os.environ["OMP_NUM_THREADS"] = str(DEFAULT_PROCESSES)

# Append transBin to sys.path to allow transBin import even if transBin was not installed
# using pip
parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parentdir)

import numpy as np
import transBin


################################# DEFINE FUNCTIONS ##########################
def log(string, logfile, indent=0):
    print(('\t' * indent) + string, file=logfile)
    logfile.flush()


def calc_tnf(outdir, fastapath, tnfpath, namespath, lengthspath, mincontiglength, logfile):
    begintime = time.time()
    log('\nLoading TNF', logfile, 0)
    log('Minimum sequence length: {}'.format(mincontiglength), logfile, 1)
    # If no path to FASTA is given, we load TNF from .npz files
    if fastapath is None:
        log('Loading TNF from npz array {}'.format(tnfpath), logfile, 1)
        tnfs = transBin.transBintools.read_npz(tnfpath)
        log('Loading contignames from npz array {}'.format(namespath), logfile, 1)
        contignames = transBin.transBintools.read_npz(namespath)
        log('Loading contiglengths from npz array {}'.format(lengthspath), logfile, 1)
        contiglengths = transBin.transBintools.read_npz(lengthspath)

        if not tnfs.dtype == np.float32:
            raise ValueError('TNFs .npz array must be of float32 dtype')

        if not np.issubdtype(contiglengths.dtype, np.integer):
            raise ValueError('contig lengths .npz array must be of an integer dtype')

        if not (len(tnfs) == len(contignames) == len(contiglengths)):
            raise ValueError('Not all of TNFs, names and lengths are same length')

        # Discard any sequence with a length below mincontiglength
        mask = contiglengths >= mincontiglength
        tnfs = tnfs[mask]
        contignames = list(contignames[mask])
        contiglengths = contiglengths[mask]

    # Else parse FASTA files
    else:
        log('Loading data from FASTA file {}'.format(fastapath), logfile, 1)
        with transBin.transBintools.Reader(fastapath, 'rb') as tnffile:
            ret = transBin.parsecontigs.read_contigs(tnffile, minlength=mincontiglength)

        tnfs, contignames, contiglengths = ret
        transBin.transBintools.write_npz(os.path.join(outdir, 'tnf.npz'), tnfs)
        transBin.transBintools.write_npz(os.path.join(outdir, 'lengths.npz'), contiglengths)

    elapsed = round(time.time() - begintime, 2)
    ncontigs = len(contiglengths)
    nbases = contiglengths.sum()

    print('', file=logfile)
    log('Kept {} bases in {} sequences'.format(nbases, ncontigs), logfile, 1)
    log('Processed TNF in {} seconds'.format(elapsed), logfile, 1)

    return tnfs, contignames, contiglengths


def calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash, ncontigs,
              minalignscore, minid, subprocesses, logfile):
    begintime = time.time()
    log('\nLoading RPKM', logfile)
    # If rpkm is given, we load directly from .npz file
    if rpkmpath is not None:
        log('Loading RPKM from npz array {}'.format(rpkmpath), logfile, 1)
        rpkms = transBin.transBintools.read_npz(rpkmpath)

        if not rpkms.dtype == np.float32:
            raise ValueError('RPKMs .npz array must be of float32 dtype')

    else:
        log('Reference hash: {}'.format(refhash if refhash is None else refhash.hex()), logfile, 1)

    # Else if JGI is given, we load from that
    if jgipath is not None:
        log('Loading RPKM from JGI file {}'.format(jgipath), logfile, 1)
        with open(jgipath) as file:
            rpkms = transBin.transBintools._load_jgi(file, mincontiglength, refhash)

    elif bampaths is not None:
        log('Parsing {} BAM files with {} subprocesses'.format(len(bampaths), subprocesses),
            logfile, 1)
        log('Min alignment score: {}'.format(minalignscore), logfile, 1)
        log('Min identity: {}'.format(minid), logfile, 1)
        log('Min contig length: {}'.format(mincontiglength), logfile, 1)
        log('\nOrder of columns is:', logfile, 1)
        log('\n\t'.join(bampaths), logfile, 1)
        print('', file=logfile)

        dumpdirectory = os.path.join(outdir, 'tmp')
        rpkms = transBin.parsebam.read_bamfiles(bampaths, dumpdirectory=dumpdirectory,
                                            refhash=refhash, minscore=minalignscore,
                                            minlength=mincontiglength, minid=minid,
                                            subprocesses=subprocesses, logfile=logfile)
        print('', file=logfile)
        transBin.transBintools.write_npz(os.path.join(outdir, 'rpkm.npz'), rpkms)
        shutil.rmtree(dumpdirectory)

    if len(rpkms) != ncontigs:
        raise ValueError("Length of TNFs and length of RPKM does not match. Verify the inputs")

    elapsed = round(time.time() - begintime, 2)
    log('Processed RPKM in {} seconds'.format(elapsed), logfile, 1)

    return rpkms


def traintransformer(outdir, rpkms, tnfs, nlayers, dmodel, dff, nheads, dk, nlatent, alpha, beta, dropout, cuda,
             batchsize, nepochs, lrate, logfile):
    begintime = time.time()
    log('\nCreating and training Transformer', logfile)

    nsamples = rpkms.shape[1]
    transformer = transBin.transformer.Transformer(nsamples, nlatent=nlatent, d_model = dmodel, d_ff = dff, d_k = dk, nlayers = nlayers, nheads=nheads,
                          alpha=alpha, beta=beta, dropout=dropout, cuda=cuda)

    log('Created transformer', logfile, 1)
    dataloader, mask = transBin.transformer.make_dataloader(rpkms, tnfs, batchsize,
                                                   destroy=True, cuda=cuda)
    log('Created dataloader and mask', logfile, 1)
    transBin.transBintools.write_npz(os.path.join(outdir, 'mask.npz'), mask)
    n_discarded = len(mask) - mask.sum()
    log('Number of sequences unsuitable for encoding: {}'.format(n_discarded), logfile, 1)
    log('Number of sequences remaining: {}'.format(len(mask) - n_discarded), logfile, 1)
    print('', file=logfile)

    modelpath = os.path.join(outdir, 'model.pt')
    transformer.trainmodel(dataloader, nepochs=nepochs, lrate=lrate,
                   logfile=logfile, modelfile=modelpath)

    print('', file=logfile)
    log('Encoding to latent representation', logfile, 1)
    latent = transformer.encode(dataloader)
    transBin.transBintools.write_npz(os.path.join(outdir, 'latent.npz'), latent)
    del transformer  # Needed to free "latent" array's memory references?

    elapsed = round(time.time() - begintime, 2)
    log('Trained transformer and encoded in {} seconds'.format(elapsed), logfile, 1)

    return mask, latent


def cluster(clusterspath, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, separator, cuda, logfile):
    begintime = time.time()

    log('\nClustering', logfile)
    log('Windowsize: {}'.format(windowsize), logfile, 1)
    log('Min successful thresholds detected: {}'.format(minsuccesses), logfile, 1)
    log('Max clusters: {}'.format(maxclusters), logfile, 1)
    log('Min cluster size: {}'.format(minclustersize), logfile, 1)
    log('Use CUDA for clustering: {}'.format(cuda), logfile, 1)
    log('Separator: {}'.format(None if separator is None else ('"' + separator + '"')),
        logfile, 1)

    it = transBin.cluster.cluster(latent, contignames, destroy=True, windowsize=windowsize,
                              normalized=False, minsuccesses=minsuccesses, cuda=cuda)

    renamed = ((str(i + 1), c) for (i, (n, c)) in enumerate(it))

    # Binsplit if given a separator
    if separator is not None:
        renamed = transBin.transBintools.binsplit(renamed, separator)

    with open(clusterspath, 'w') as clustersfile:
        _ = transBin.transBintools.write_clusters(clustersfile, renamed, max_clusters=maxclusters,
                                          min_size=minclustersize, rename=False)
    clusternumber, ncontigs = _

    print('', file=logfile)
    log('Clustered {} contigs in {} bins'.format(ncontigs, clusternumber), logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log('Clustered contigs in {} seconds'.format(elapsed), logfile, 1)


def write_fasta(outdir, clusterspath, fastapath, contignames, contiglengths, minfasta, logfile):
    begintime = time.time()

    log('\nWriting FASTA files', logfile)
    log('Minimum FASTA size: {}'.format(minfasta), logfile, 1)

    lengthof = dict(zip(contignames, contiglengths))
    filtered_clusters = dict()

    with open(clusterspath) as file:
        clusters = transBin.transBintools.read_clusters(file)

    for cluster, contigs in clusters.items():
        size = sum(lengthof[contig] for contig in contigs)
        if size >= minfasta:
            filtered_clusters[cluster] = clusters[cluster]

    del lengthof, clusters
    keep = set()
    for contigs in filtered_clusters.values():
        keep.update(set(contigs))

    with transBin.transBintools.Reader(fastapath, 'rb') as file:
        fastadict = transBin.transBintools.loadfasta(file, keep=keep)

    transBin.transBintools.write_bins(os.path.join(outdir, "bins"), filtered_clusters, fastadict, maxbins=None)

    ncontigs = sum(map(len, filtered_clusters.values()))
    nfiles = len(filtered_clusters)
    print('', file=logfile)
    log('Wrote {} contigs to {} FASTA files'.format(ncontigs, nfiles), logfile, 1)

    elapsed = round(time.time() - begintime, 2)
    log('Wrote FASTA in {} seconds'.format(elapsed), logfile, 1)


def run(outdir, fastapath, tnfpath, namespath, lengthspath, bampaths, rpkmpath, jgipath,
        mincontiglength, norefcheck, minalignscore, minid, subprocesses, nlatent,
        nlayers, dmodel, dff, nheads, dk, nepochs, batchsize, cuda, alpha, beta, dropout, lrate, windowsize,
        minsuccesses, minclustersize, separator, maxclusters, minfasta, logfile):
    log('Starting transBin version ' + '.'.join(map(str, transBin.__version__)), logfile)
    log('Date and time is ' + str(datetime.datetime.now()), logfile, 1)
    begintime = time.time()

    # Get TNFs, save as npz
    tnfs, contignames, contiglengths = calc_tnf(outdir, fastapath, tnfpath, namespath,
                                                lengthspath, mincontiglength, logfile)

    # Parse BAMs, save as npz
    refhash = None if norefcheck else transBin.transBintools._hash_refnames(
        (name.split(maxsplit=1)[0] for name in contignames)
    )
    rpkms = calc_rpkm(outdir, bampaths, rpkmpath, jgipath, mincontiglength, refhash,
                      len(tnfs), minalignscore, minid, subprocesses, logfile)

    # Train, save model
    mask, latent = traintransformer(outdir, rpkms, tnfs, nlayers, dmodel, dff, nheads, dk, nlatent, alpha, beta,
                            dropout, cuda, batchsize, nepochs, lrate, logfile)

    del tnfs, rpkms
    contignames = [c for c, m in zip(contignames, mask) if m]

    # Cluster, save tsv file
    clusterspath = os.path.join(outdir, 'clusters.tsv')
    cluster(clusterspath, latent, contignames, windowsize, minsuccesses, maxclusters,
            minclustersize, separator, cuda, logfile)

    del latent

    if minfasta is not None:
        write_fasta(outdir, clusterspath, fastapath, contignames, contiglengths, minfasta,
                    logfile)

    elapsed = round(time.time() - begintime, 2)
    log('\nCompleted transBin in {} seconds'.format(elapsed), logfile)

    # TODO:获取模型评价指标值,并return ,copy from cmd_benchmark.py
    # Check that files exist
    refpath = 'data/gi/reference.tsv'
    taxpath = 'data/gi/taxonomy.tsv'
    for path in clusterspath, refpath, taxpath:
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError(path)

    with open(clusterspath) as file:
        clusters = transBin.transBintools.read_clusters(file)

    with open(refpath) as file:
        reference = transBin.benchmark.Reference.from_file(file)

    if taxpath is not None:
        with open(taxpath) as file:
            reference.load_tax_file(file)

    binning = transBin.benchmark.Binning(clusters, reference, minsize=2000, disjoint=None,
                                     binsplit_separator=separator)

    score = 0
    _score = 0
    for rank in range(len(binning.counters)):
        binning.print_matrix(rank)
        binning.print_matrix(rank,file=logfile)
        _score = binning.counters[rank][(0.90, 0.95)]
        score = score + _score
        print("_score", rank, '=', _score, file=logfile)
    print('score= ', score, file=logfile)
    return score


def objective(trial, args):
    outdir = os.path.join(args.outdir, 'trial_' + str(trial.number))
    os.mkdir(outdir)
    logpath = os.path.join(outdir, 'log.txt')

    # TODO:不考虑  nhiddens,nepochs, cuda,lrate
    # 考虑 nhiddens,nlatent,batchsize, alpha, beta, dropout
    # nlatent = trial.suggest_categorical('nlatent', [32, 64, 128, 256])
    nlatent = trial.suggest_categorical('nlatent', [32])
    # batchsize = trial.suggest_categorical('batchsize', [128,256, 512])
    batchsize = trial.suggest_categorical('batchsize', [256])
    # batchsize = args.batchsize
    # alpha不能取到1.0
    # alpha = trial.suggest_float('alpha', 0.1, 0.5, step=0.10)
    # alpha = trial.suggest_categorical('alpha', [0.15])
    # alpha = args.alpha
    # beta = trial.suggest_float('beta', 200, 2000, step=600)
    beta = args.beta
    # dropout = trial.suggest_float('dropout', 0, 0.4, step=0.20)
    dropout = args.dropout
    # nlayers = trial.suggest_categorical('nlayers', [1,2,4,6])
    nlayers = args.nlayers
    ###################### SET UP LAST PARAMS ############################

    # This doesn't actually work, but maybe the PyTorch folks will fix it sometime.
    subprocesses = args.subprocesses
    torch.set_num_threads(args.subprocesses)
    if args.bamfiles is not None:
        subprocesses = min(subprocesses, len(args.bamfiles))

    # TODO:注意 nlatent=nlatent,而非args.nlatent;batchsize=...
    with open(logpath, 'w') as logfile:
        return run(outdir,
                    args.fasta,
                    args.tnfs,
                    args.names,
                    args.lengths,
                    args.bamfiles,
                    args.rpkm,
                    args.jgi,
                    mincontiglength=args.minlength,
                    norefcheck=args.norefcheck,
                    minalignscore=args.minascore,
                    minid=args.minid,
                    subprocesses=subprocesses,
                    nlatent=nlatent,
                    nlayers=nlayers,
                    dmodel=args.dmodel,
                    dff=args.dff,
                    nheads=args.nheads,
                    dk=args.dk,
                    nepochs=args.nepochs,
                    batchsize=batchsize,
                    cuda=args.cuda,
                    alpha=args.alpha,
                    beta=beta,
                    dropout=dropout,
                    lrate=args.lrate,
                    windowsize=args.windowsize,
                    minsuccesses=args.minsuccesses,
                    minclustersize=args.minsize,
                    separator=args.separator,
                    maxclusters=args.maxclusters,
                    minfasta=args.minfasta,
                    logfile=logfile)

def main():
    doc = f"""transBin: Variational autoencoders for metagenomic binning.

    Version: {'.'.join([str(i) for i in transBin.__version__])}

    Default use, good for most datasets:
    transBin --outdir out --fasta my_contigs.fna --bamfiles *.bam

    For advanced use and extensions of transBin, check documentation of the package
    at https://github.com/RasmussenLab/transBin."""
    parser = argparse.ArgumentParser(
        prog="transBin",
        description=doc,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        usage="%(prog)s outdir tnf_input rpkm_input [options]",
        add_help=False)

    # Help
    helpos = parser.add_argument_group(title='Help', description=None)
    helpos.add_argument('-h', '--help', help='print help and exit', action='help')

    # Positional arguments
    reqos = parser.add_argument_group(title='Output (required)', description=None)
    reqos.add_argument('--outdir', metavar='', required=True, help='output directory to create')

    # TNF arguments
    tnfos = parser.add_argument_group(title='TNF input (either fasta or all .npz files required)')
    tnfos.add_argument('--fasta', metavar='', help='path to fasta file')
    tnfos.add_argument('--tnfs', metavar='', help='path to .npz of TNF')
    tnfos.add_argument('--names', metavar='', help='path to .npz of names of sequences')
    tnfos.add_argument('--lengths', metavar='', help='path to .npz of seq lengths')

    # RPKM arguments
    rpkmos = parser.add_argument_group(title='RPKM input (either BAMs, JGI or .npz required)')
    rpkmos.add_argument('--bamfiles', metavar='', help='paths to (multiple) BAM files', nargs='+')
    rpkmos.add_argument('--rpkm', metavar='', help='path to .npz of RPKM')
    rpkmos.add_argument('--jgi', metavar='', help='path to output of jgi_summarize_bam_contig_depths')

    # Optional arguments
    inputos = parser.add_argument_group(title='IO options', description=None)

    inputos.add_argument('-m', dest='minlength', metavar='', type=int, default=100,
                         help='ignore contigs shorter than this [100]')
    inputos.add_argument('-s', dest='minascore', metavar='', type=int, default=None,
                         help='ignore reads with alignment score below this [None]')
    inputos.add_argument('-z', dest='minid', metavar='', type=float, default=None,
                         help='ignore reads with nucleotide identity below this [None]')
    inputos.add_argument('-p', dest='subprocesses', metavar='', type=int, default=DEFAULT_PROCESSES,
                         help=('number of subprocesses to spawn '
                               '[min(' + str(DEFAULT_PROCESSES) + ', nbamfiles)]'))
    inputos.add_argument('--norefcheck', help='skip reference name hashing check [False]',
                         action='store_true')
    inputos.add_argument('--minfasta', dest='minfasta', metavar='', type=int, default=None,
                         help='minimum bin size to output as fasta [None = no files]')

    # VAE arguments
    transos = parser.add_argument_group(title='Transformer options', description=None)

    transos.add_argument('-l', dest='nlatent', metavar='', type=int,
                       default=32, help='latent neurons [32]')
    transos.add_argument('-a', dest='alpha', metavar='', type=float,
                       default=0.15, help='alpha, weight of TNF versus depth loss [Auto]')
    transos.add_argument('-b', dest='beta', metavar='', type=float,
                       default=200.0, help='beta, capacity to learn [200.0]')
    transos.add_argument('-d', dest='dropout', metavar='', type=float,
                       default=None, help='dropout [Auto]')
    transos.add_argument('-dmodel', dest='dmodel', metavar='', type=int,
                         default=512, help='transformer d_model')
    transos.add_argument('-dff', dest='dff', metavar='', type=int,
                         default=2048, help='transformer d_ff')
    transos.add_argument('-nlayers', dest='nlayers', metavar='', type=int,
                         default=4, help='transformer nlayers')
    transos.add_argument('-nheads', dest='nheads', metavar='', type=int,
                         default=8, help='transformer nheads')
    transos.add_argument('-dk', dest='dk', metavar='', type=int,
                         default=64, help='transformer dk')
    transos.add_argument('--cuda', help='use GPU to train & cluster [False]', action='store_true')

    trainos = parser.add_argument_group(title='Training options', description=None)

    trainos.add_argument('-e', dest='nepochs', metavar='', type=int,
                         default=500, help='epochs [500]')
    trainos.add_argument('-t', dest='batchsize', metavar='', type=int,
                         default=256, help='starting batch size [256]')
    trainos.add_argument('-r', dest='lrate', metavar='', type=float,
                         default=1e-3, help='learning rate [0.001]')

    # Clustering arguments
    clusto = parser.add_argument_group(title='Clustering options', description=None)
    clusto.add_argument('-w', dest='windowsize', metavar='', type=int,
                        default=200, help='size of window to count successes [200]')
    clusto.add_argument('-u', dest='minsuccesses', metavar='', type=int,
                        default=20, help='minimum success in window [20]')
    clusto.add_argument('-i', dest='minsize', metavar='', type=int,
                        default=1, help='minimum cluster size [1]')
    clusto.add_argument('-c', dest='maxclusters', metavar='', type=int,
                        default=None, help='stop after c clusters [None = infinite]')
    clusto.add_argument('-o', dest='separator', metavar='', type=str,
                        default=None, help='binsplit separator [None = no split]')

    ######################### PRINT HELP IF NO ARGUMENTS ###################
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit()

    args = parser.parse_args()
    ######################### CHECK INPUT/OUTPUT FILES #####################

    # Outdir does not exist
    args.outdir = os.path.abspath(args.outdir)
    if os.path.exists(args.outdir):
        raise FileExistsError(args.outdir)

    # Outdir is in an existing parent dir
    parentdir = os.path.dirname(args.outdir)
    if parentdir and not os.path.isdir(parentdir):
        raise NotADirectoryError(parentdir)

    # Make sure only one TNF input is there
    if args.fasta is None:
        for path in args.tnfs, args.names, args.lengths:
            if path is None:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the three .npz inputs')
            if not os.path.isfile(path):
                raise FileNotFoundError(path)
    else:
        for path in args.tnfs, args.names, args.lengths:
            if path is not None:
                raise argparse.ArgumentTypeError('Must specify either FASTA or the three .npz inputs')
        if not os.path.isfile(args.fasta):
            raise FileNotFoundError('Not an existing non-directory file: ' + args.fasta)

    if args.dmodel != args.nheads*args.dk:
        raise argparse.ArgumentTypeError('Dmodel must be equal to nheads * dk')

    # Make sure only one RPKM input is there
    if sum(i is not None for i in (args.bamfiles, args.rpkm, args.jgi)) != 1:
        raise argparse.ArgumentTypeError('Must specify exactly one of BAM files, JGI file or RPKM input')

    for path in args.rpkm, args.jgi:
        if path is not None and not os.path.isfile(path):
            raise FileNotFoundError('Not an existing non-directory file: ' + args.rpkm)

    if args.bamfiles is not None:
        for bampath in args.bamfiles:
            if not os.path.isfile(bampath):
                raise FileNotFoundError('Not an existing non-directory file: ' + bampath)

    # Check minfasta settings
    if args.minfasta is not None and args.fasta is None:
        raise argparse.ArgumentTypeError('If minfasta is not None, '
                                         'input fasta file must be given explicitly')

    if args.minfasta is not None and args.minfasta < 0:
        raise argparse.ArgumentTypeError('Minimum FASTA output size must be nonnegative')

    ####################### CHECK ARGUMENTS FOR TNF AND BAMFILES ###########
    if args.minlength < 100:
        raise argparse.ArgumentTypeError('Minimum contig length must be at least 100')

    if args.minid is not None and (args.minid < 0 or args.minid >= 1.0):
        raise argparse.ArgumentTypeError('Minimum nucleotide ID must be in [0,1)')

    if args.minid is not None and args.bamfiles is None:
        raise argparse.ArgumentTypeError('If minid is set, RPKM must be passed as bam files')

    if args.minascore is not None and args.bamfiles is None:
        raise argparse.ArgumentTypeError('If minascore is set, RPKM must be passed as bam files')

    if args.subprocesses < 1:
        raise argparse.ArgumentTypeError('Zero or negative subprocesses requested')

    ####################### CHECK VAE OPTIONS ################################

    if args.nlatent < 1:
        raise argparse.ArgumentTypeError('Minimum 1 latent neuron, not {}'.format(args.latent))

    if args.alpha is not None and (args.alpha <= 0 or args.alpha >= 1):
        raise argparse.ArgumentTypeError('alpha must be above 0 and below 1')

    if args.beta <= 0:
        raise argparse.ArgumentTypeError('beta cannot be negative or zero')

    if args.dropout is not None and (args.dropout < 0 or args.dropout >= 1):
        raise argparse.ArgumentTypeError('dropout must be in 0 <= d < 1')

    if args.cuda and not torch.cuda.is_available():
        raise ModuleNotFoundError('Cuda is not available on your PyTorch installation')

    ###################### CHECK TRAINING OPTIONS ####################
    if args.nepochs < 1:
        raise argparse.ArgumentTypeError('Minimum 1 epoch, not {}'.format(args.nepochs))

    if args.batchsize < 1:
        raise argparse.ArgumentTypeError('Minimum batchsize of 1, not {}'.format(args.batchsize))

    if args.lrate <= 0:
        raise argparse.ArgumentTypeError('Learning rate must be positive')

    ###################### CHECK CLUSTERING OPTIONS ####################
    if args.minsize < 1:
        raise argparse.ArgumentTypeError('Minimum cluster size must be at least 0')

    if args.windowsize < 1:
        raise argparse.ArgumentTypeError('Window size must be at least 1')

    if args.minsuccesses < 1 or args.minsuccesses > args.windowsize:
        raise argparse.ArgumentTypeError('Minimum cluster size must be in 1:windowsize')

    if args.separator is not None and len(args.separator) == 0:
        raise argparse.ArgumentTypeError('Binsplit separator cannot be an empty string')

    ###################### SET UP LAST PARAMS ############################

    ################### RUN PROGRAM #########################
    if not os.path.exists(args.outdir):
        os.mkdir(args.outdir)

    ################### RUN Trial and study #########################
    # Create a study object and optimize the objective function
    # sampler=optuna.samplers.RandomSampler() or use default(TPE)
    study = optuna.create_study(direction='maximize')
    # study = optuna.create_study(study_name="distributed-example",storage="mysql://root:123456@localhost/example",load_if_exists=True,direction='maximize',sampler=optuna.samplers.RandomSampler())
    study.optimize(lambda trial: objective(trial, args), n_trials=30)

    # Print the best hyperparameters and objective value
    with open(os.path.join(args.outdir, 'superspace.txt'), "w") as f:
        for _trial in study.trials:
            print("Trial parameters: {}".format(_trial.params), file=f)
            print("Trial value: {}".format(_trial.value), file=f)
        print('Best GC genome numbers: {}'.format(study.best_value), file=f)
        print("Best hyperparameters: {}".format(study.best_params), file=f)

    fig1 = optuna.visualization.plot_parallel_coordinate(study)
    fig1.write_image(os.path.join(args.outdir, 'superspace_parallel.png'))
    fig2 = optuna.visualization.plot_param_importances(study)
    fig2.write_image(os.path.join(args.outdir, 'superspace_importance.png'))


if __name__ == '__main__':
    main()
