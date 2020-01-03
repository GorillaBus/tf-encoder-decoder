import tensorflow as tf
from sklearn.model_selection import train_test_split
import unicodedata
import re
import io

'''
	Handles data loading, pre-processing and tokenizing

'''
class DataHandler():

	'''
		Creates a tf.data.Dataset object to feed the network
		with tensors and in batches

		X:			  	(tensor) with input data
		y:			  	(tensor) with data labels (X paired)
		batch_size:	 	(int) representing desired batch size

		returns:		(tf.data.Dataset) dataset object
						(int) length of the dataset

	'''
	def create_tf_dataset(self, X, y, batch_size=64):
		data_length = len(X)
		dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(data_length)
		dataset = dataset.batch(batch_size, drop_remainder=True)
		return dataset, data_length



	'''
		Loads data from a file with the following format:

		<sentence_1_lang_A>	<tab> <sentence_1_lang_B>
		<sentence_2_lang_A>	<tab> <sentence_2_lang_B>
		<sentence_3_lang_A>	<tab> <sentence_3_lang_B>
		...
		<sentence_N_lang_A>	<tab> <sentence_N_lang_B>

		path:			(string) path to the dataset file
		num_examples:	(int) 0 for full load or N to limit max load

	'''
	def load_from_file(self, path, num_examples=0):
	    lines = io.open(path, encoding='UTF-8').read().strip().split('\n')
	    if num_examples == 0:
	        num_examples = len(lines)
	    word_pairs = [[ self.process_sentence(w) for w in l.split('\t')]  for l in lines[:num_examples]]
	    return zip(*word_pairs)




	'''
		Splits a dataset represented by X, y data pairs by a factor

		X:				(tensor) with input data
		y:				(tensor) with data labels (X paired)
		split_size:		(float) number between 0 and 1

		returns:		(tensor) splitted X, y pairs

	'''
	def split_data(self, X, y, split_size=0.25):
		X, X_test, y, y_test = train_test_split(X, y, test_size=split_size)
		return X, X_test, y, y_test



	'''
		Given a tensor of sentences, returns the maximun number of words found
		on a sentence

		tensor:		(tensor) an array of sentences where each sentence is a tensor

		returns:	(int) max number of words found

	'''
	def max_length(self, tensor):
	    return max(len(t) for t in tensor)



	'''
		Cleans a given text by removing accents and unwanted characters
		and adds start/stop tokens

	'''
	def process_sentence(self, w):
	    w = self.unicode_to_ascii(w.lower().strip())

	    # Creates a space between a word and the punctuation following it
	    w = re.sub(r"([?.!,¿])", r" \1 ", w)
	    w = re.sub(r'[" "]+', " ", w)

	    # Replace everything with a space except (a-z, A-Z, ".", "?", "!", ",")
	    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)
	    w = w.rstrip().strip()

	    w = '<start> ' + w + ' <end>'
	    return w



	'''
		Creates a vocbulary for a given text
		Returns a list of word_id sequence for each sentence and a tokenizer

	'''
	def tokenize(self, text):
	    vocab_tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='')
	    vocab_tokenizer.fit_on_texts(text)
	    tensor = vocab_tokenizer.texts_to_sequences(text)
	    tensor = tf.keras.preprocessing.sequence.pad_sequences(tensor, padding='post')
	    return tensor, vocab_tokenizer



	def unicode_to_ascii(self, s):
	    return ''.join(c for c in unicodedata.normalize('NFD', s)
	        if unicodedata.category(c) != 'Mn')
