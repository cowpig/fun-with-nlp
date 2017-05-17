# http://mattmahoney.net/dc/text8.zip
from utils import build_dataset

fn = 'data/text8.txt'
with open(fn) as f:
	words = f.read().strip().split()

vocabulary_size = 50000


data, count, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size)
