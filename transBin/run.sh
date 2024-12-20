#!/usr/bin/env bash
set -ex

# This is the master script for the capsule. When you click "Reproducible Run", the code in this file will execute.

# install transBin and dependencies
pip install -e /home/neu/xcy/TransBin
cd ..

# using GPU for acceleration
# important: it can run with only CPUs, then it is ~1 hour per dataset
# to run without GPU remove "--cuda"

# Note that output may vary between runs due to the stochastic process of training the neural networks

# run on CAMI2 datasets
# using multi-split (as here we had individual assemblies available)
#transBin --outdir results/airways --fasta data/airways/contigs.fna.gz --rpkm data/airways/abundance.npz -o C --cuda
#transBin --outdir /home/neu/xcy/TransBin/results/gi --fasta /home/neu/xcy/TransBin/data/gi/contigs.fna.gz --rpkm /home/neu/xcy/TransBin/data/gi/abundance.npz -o C --cuda
#transBin --outdir results/oral --fasta data/oral/contigs.fna.gz --rpkm data/oral/abundance.npz -o C --cuda
#transBin --outdir results/skin --fasta data/skin/contigs.fna.gz --rpkm data/skin/abundance.npz -o C --cuda
transBin --outdir /home/neu/xcy/TransBin/results/urog --fasta /home/neu/xcy/TransBin/data/urog/contigs.fna.gz --rpkm /home/neu/xcy/TransBin/data/urog/abundance.npz -o C --cuda

# run on MetaHIT dataset
# without multi-split (as here we had pooled assemblies)
#transBin --outdir results/metahit --fasta data/metahit/contigs.fna.gz --rpkm data/metahit/abundance.npz --cuda

# benchmark
python3 transBin/src/cmd_benchmark.py --tax data/airways/taxonomy.tsv transBin results/airways/clusters.tsv data/airways/reference.tsv > results/airways/benchmark.tsv

python3 transBin/src/cmd_benchmark.py --tax data/gi/taxonomy.tsv transBin results/gi/clusters.tsv data/gi/reference.tsv > results/gi/benchmark.tsv

python3 transBin/src/cmd_benchmark.py --tax data/oral/taxonomy.tsv transBin results/oral/clusters.tsv data/oral/reference.tsv > results/oral/benchmark.tsv

python3 transBin/src/cmd_benchmark.py --tax data/skin/taxonomy.tsv transBin results/skin/clusters.tsv data/skin/reference.tsv > results/skin/benchmark.tsv

python3 transBin/src/cmd_benchmark.py --tax data/urog/taxonomy.tsv transBin results/urog/clusters.tsv data/urog/reference.tsv > results/urog/benchmark.tsv

python3 transBin/src/cmd_benchmark.py --tax data/metahit/taxonomy.tsv transBin results/metahit/clusters.tsv data/metahit/reference.tsv > results/metahit/benchmark.tsv

# print benchmarks to screen
head -n 10000 results/*/benchmark.tsv