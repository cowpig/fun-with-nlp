# http://mattmahoney.net/dc/text8.zip
import argparse
from utils import build_dataset

parser = argparse.ArgumentParser()
parser.add_argument('--fn', help='path to file', type=str)
args = parser.parse_args()

fn = args.fn
with open(fn) as f:
    words = f.read().strip().split()

vocabulary_size = 50000


data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
