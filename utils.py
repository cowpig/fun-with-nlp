import tensorflow as tf
import numpy as np
from collections import Counter

def build_dataset(words, vocabulary_size):
	count = [['UNK', -1]]
	count.extend(Counter(words).most_common(vocabulary_size - 1))
	dictionary = dict()

	for word, _ in count:
		dictionary[word] = len(dictionary)
	
	data = list()
	unk_count = 0

	for word in words:
		if word in dictionary:
			index = dictionary[word]
		else:
			index = 0  # dictionary['UNK']
			unk_count += 1

		data.append(index)
	
	count[0][1] = unk_count
	reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
	return data, count, dictionary, reverse_dictionary

def generate_batch(data, idx, batch_size, num_skips, skip_window):
	assert batch_size % num_skips == 0
	assert num_skips <= 2 * skip_window

	batch = np.ndarray(shape=(batch_size), dtype=np.int32)
	labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
	span = 2 * skip_window + 1  # [ skip_window target skip_window ]
	buf = collections.deque(maxlen=span)

	for _ in range(span):
		buf.append(data[idx])
		idx = (idx + 1) % len(data)
	
	for i in range(batch_size // num_skips):
		target = skip_window  # target label at the center of the buffer
		targets_to_avoid = [skip_window]
		for j in range(num_skips):
			while target in targets_to_avoid:
				target = random.randint(0, span - 1)
			targets_to_avoid.append(target)
			batch[i * num_skips + j] = buf[skip_window]
			labels[i * num_skips + j, 0] = buf[target]
		buf.append(data[idx])
		idx = (idx + 1) % len(data)
	
	# Backtrack a little bit to avoid skipping words in the end of a batch
	idx = (idx + len(data) - span) % len(data)
	return batch, labels
