import tensorflow as tf
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
import time
import sys
import os
import pyter



'''
	Encoder

	input_size:	 equals to vocabulary size
	hidden_dim:	 equals to embedding dimmension

'''
class Encoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, units, dropout_rate=0.0):

		super(Encoder, self).__init__()
		self.units = units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,
									   return_sequences=True,
									   return_state=True,
									   dropout=dropout_rate,
									   recurrent_dropout=0,						# Forced to 0 for CuDNN_GRU
									   recurrent_initializer='glorot_uniform')

	@tf.function
	def call(self, x, hidden, training=False):
		x = self.embedding(x)
		output, state = self.gru(x, initial_state=hidden, training=training)
		return output, state

	def initialize_hidden_state(self, batch_size):
		return tf.zeros((batch_size, self.units))


'''
	Attention module

'''
class BahdanauAttention(tf.keras.layers.Layer):
	def __init__(self, units):
		super(BahdanauAttention, self).__init__()
		self.W1 = tf.keras.layers.Dense(units)
		self.W2 = tf.keras.layers.Dense(units)
		self.V = tf.keras.layers.Dense(1)

	@tf.function
	def call(self, query, values):
		# hidden shape == (batch_size, hidden size)
		# hidden_with_time_axis shape == (batch_size, 1, hidden size)
		# we are doing this to perform addition to calculate the score
		hidden_with_time_axis = tf.expand_dims(query, 1)

		# score shape == (batch_size, max_length, 1)
		# we get 1 at the last axis because we are applying score to self.V
		# the shape of the tensor before applying self.V is (batch_size, max_length, units)
		score = self.V(tf.nn.tanh(
			self.W1(values) + self.W2(hidden_with_time_axis)))

		# attention_weights shape == (batch_size, max_length, 1)
		attention_weights = tf.nn.softmax(score, axis=1)

		# context_vector shape after sum == (batch_size, hidden_size)
		context_vector = attention_weights * values
		context_vector = tf.reduce_sum(context_vector, axis=1)

		return context_vector, attention_weights


'''
	Decoder

'''
class Decoder(tf.keras.Model):
	def __init__(self, vocab_size, embedding_dim, dec_units, dropout_rate=0.0):
		super(Decoder, self).__init__()

		self.units = dec_units
		self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
		self.gru = tf.keras.layers.GRU(self.units,
									   return_sequences=True,
									   return_state=True,
									   dropout=dropout_rate,
									   recurrent_dropout=0,						# Forced to 0 for CuDNN_GRU
									   recurrent_initializer='glorot_uniform')

		self.fc = tf.keras.layers.Dense(vocab_size)

		# used for attention
		self.attention = BahdanauAttention(self.units)
		#self.attention = tf.keras.layers.Attention(self.units)

	@tf.function
	def call(self, x, hidden, enc_output, training=False):
		# enc_output shape == (batch_size, max_length, hidden_size)
		context_vector, attention_weights = self.attention(hidden, enc_output)

		# x shape after passing through embedding == (batch_size, 1, embedding_dim)
		x = self.embedding(x)

		# x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
		x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

		# passing the concatenated vector to the GRU
		output, state = self.gru(x, training=training)

		# output shape == (batch_size * 1, hidden_size)
		output = tf.reshape(output, (-1, output.shape[2]))

		# output shape == (batch_size, vocab)
		x = self.fc(output)

		return x, state, attention_weights



'''
Encoder Decoder Model Wrapper

'''
class EncoderDecoderWrapper():

	'''
	Class constructor

	enc_vocab_size:			 (int) encoder vocabulary size
	dec_vocab_size:			 (int) decoder vocabulary size
	embedding_dim:			 (int) embedding layer input size
	output_dim:		  		 (int) max decoder sequence tokens (network output)
	units:					 (int) desired number of GRU units for both layers
	dropout_rate:			 (float) dropout rate to in GRU units
	checkpoint_path:		 (string) path where to save checkpoints

	'''
	def __init__(self,
				enc_vocab_size=0,
				dec_vocab_size=0,
				embedding_dim=0,
				input_dim=0,
				output_dim=0,
				units=64,
				dropout_rate=0.0,
				checkpoint_path='./model_checkpoints'):

		# Check inputs
		if (enc_vocab_size < 1 ) | (dec_vocab_size < 1):
			print('enc_vocab_size / dec_vocab_size are required')
			return False

		if (embedding_dim < 1):
			print('embedding_dim is required')
			return False

		if (input_dim < 1):
			print('input_dim is required')
			return False

		if (output_dim < 1):
			print('output_dim is required')
			return False


		# Create checkpoints directory if unexistent
		if not os.path.exists(checkpoint_path):
			os.makedirs(checkpoint_path)

		# Construction
		self.input_dim = input_dim
		self.output_dim = output_dim
		self.encoder = Encoder(enc_vocab_size, embedding_dim, units, dropout_rate)
		self.decoder = Decoder(dec_vocab_size, embedding_dim, units, dropout_rate)
		self.optimizer = tf.keras.optimizers.Adam()
		self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
		self.__last_saved = 0
		self.checkpoint_prefix = os.path.join(checkpoint_path, "ckpt")


	'''
		Orchestrates a training session

		X:					(tensor) with input data
		y:					(tensor) with data labels (X paired)
		batch_size:			(int) representing desired batch size
		epochs:				(int) number of epochs to train for
		validation_split:   (float) multiplier to split train data into train / val,
							use 0 to skip validation
		nte_monitor:		(boolean) also report test with NO teacher enforcement on each epoch
		checkpoint_path:	Path where checkpoints should be saved
							By default everytime val loss decreases and the normalized error
							difference with train is lesser than 0.5, the model is saved

		returns:			(dict) training history with loss/nte/ter metrics per epoch

	'''
	def fit(self, X, y, batch_size=64, epochs=1, validation_split=0, nte_monitor=False):

		# Create TF datasets
		if validation_split != 0:
			X, X_val, y, y_val = train_test_split(X, y, test_size=validation_split)
			val_dataset, val_length = self.create_tf_dataset(X_val, y_val, batch_size)

		train_dataset, train_length = self.create_tf_dataset(X, y, batch_size)

		# Total batches
		steps_per_epoch = train_length // batch_size

		# Loss per epoch history
		history = {
			"loss": []
			}

		# Track minimum validation loss
		min_val_loss = 99

		# Iterate epochs and train
		for epoch in range(epochs):
			start = time.time()
			total_loss = 0

			# Get one batch (steps_per_epoch) and train from it
			for (batch, (X, y)) in enumerate(train_dataset.take(steps_per_epoch)):
				p, batch_loss = self.train_step(X, y)
				total_loss += batch_loss.numpy()

				# Report batch
				sys.stdout.write("\rEpoch: {}/{} - Batch: {}/{} - train loss: {:.4f}".format(epoch+1,
																							 epochs,
																							 batch+1,
																							 steps_per_epoch,
																							 total_loss / (batch+1)))
				sys.stdout.flush()

			# Train loss
			train_loss = total_loss / steps_per_epoch

			# Add to history
			history['loss'].append(train_loss)


			# Validation
			if validation_split != 0:
				val_loss = self._evaluate_dataset(val_dataset,
										 total_data=val_length,
										 batch_size=batch_size,
										 use_teacher_enforcement=True,  # We need a comparable value
										 verbose=False)

				# Report validation loss
				sys.stdout.write(" - val loss: {:.4f}".format(val_loss))
				sys.stdout.flush()

				# Add to history
				if 'val_loss' not in history.keys():
					history['val_loss'] = []
				history['val_loss'].append(val_loss)

				# Calculate the Normalized Error Difference
				normal_error = calc_normal_diff(train_loss, val_loss)

				# Report normal error diff
				sys.stdout.write(" - normal diff: {:.4f}".format(normal_error))

				if (min_val_loss > val_loss) & (normal_error < 0.5):
					sys.stdout.write("\n* Checkpoint: val_loss improved from {:.4f} to {:.4f}, saving...".format(min_val_loss, val_loss))
					min_val_loss = val_loss
					self.save()


			# Monitor losses with no teacher enforcement
			if nte_monitor:
				sys.stdout.write("\n* Testing train (nte)...")

				# Train loss
				nte_train_loss = self._evaluate_dataset(dataset=train_dataset,
											   total_data=train_length,
											   batch_size=batch_size,
											   use_teacher_enforcement=False,
											   verbose=False)

				# Add to history
				if 'nte_train_loss' not in history.keys():
					history['nte_train_loss'] = []
				history['nte_train_loss'].append(nte_train_loss)

				# Report
				sys.stdout.write("\r(nte) train loss: {:.4f}".format(nte_train_loss))


				# Val loss
				if validation_split != 0:
					sys.stdout.write("\r* Testing validation (nte)...")

					nte_val_loss = self._evaluate_dataset(dataset=val_dataset,
												 total_data=val_length,
												 batch_size=batch_size,
												 use_teacher_enforcement=False,
												 verbose=False)

					# Add to history
					if 'nte_val_loss' not in history.keys():
						history['nte_val_loss'] = []
					history['nte_val_loss'].append(nte_val_loss)

					# overwrite update report
					sys.stdout.write("\r(nte) train loss: {:.4f} - (nte) val loss: {:.4f}".format(nte_train_loss, nte_val_loss))
					sys.stdout.flush()

			# All done!
			sys.stdout.write("\nEpoch finished in {:.2f} secs\n".format(time.time() - start))
			sys.stdout.flush()

		return history



	'''
		Trains the network from a dataset batch or single X, y pair

		X:			tensor with input data with batch_size dimension
		y:			tensor with data labels (X paired)

		returns:	(batch) loss mean

	'''
	@tf.function
	def train_step(self, X, y):
		with tf.GradientTape() as tape:
			# Feed forward the full batch and get the mean error
			pred, loss = self.forward(X, y, training=True)

			# Compute gradient and update parameters
			variables = self.encoder.trainable_variables + self.decoder.trainable_variables
			gradients = tape.gradient(loss, variables)
			self.optimizer.apply_gradients(zip(gradients, variables))
		return pred, loss



	'''
		Test step

	'''
	@tf.function
	def test_step(self, X, y):

		# Feed forward the full batch and get the mean error
		pred, loss = self.forward(X, y, use_teacher_enforcement=False, training=False)
		return pred, loss



	'''
		Forward passes a full batch

	'''
	@tf.function
	def forward(self, X, y=None, use_teacher_enforcement=False, training=False):
		y_is_tensor = tf.is_tensor(y)

		# Initial encoder hidden state
		enc_hidden = self.encoder.initialize_hidden_state(X.shape[0])

		# Encoder forward pass
		enc_output, enc_hidden = self.encoder(X, enc_hidden, training=training)
		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([1] * X.shape[0], 1) # 1 = dictionary index for "<start>"

		# Output
		loss = 0
		batch_predictions = tf.expand_dims([1] * X.shape[0], 1)

		# Decoder forward pass
		for t in range(1, self.output_dim):
			# passing enc_output to the decoder
			predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output, training=training)

			# Get batch_predictions from the softmax output
			predicted_ids = tf.argmax(predictions, axis=1, output_type=tf.dtypes.int32)

			# Calculate loss if we have ground true values
			if y_is_tensor:
				loss += self.loss_fn(y[:, t], predictions)

				# Use teacher enforcement?
				if use_teacher_enforcement:
					dec_input = tf.expand_dims(y[:, t], 1)
				else:
					dec_input = tf.expand_dims(predicted_ids, 1)

					# Accumulate batch predictions only if NOT training
					if not training:
						batch_predictions = tf.concat([ batch_predictions, dec_input ], 1)


		# Mean loss
		if y_is_tensor:
			loss /= int(y.shape[1])

		return [batch_predictions, loss]




	'''
		Individually save weights for each model in the ensamble
		Note: checkpoints are always overwritten as a desired behaviour

	'''
	def save(self):
		self.encoder.save_weights('./model_checkpoints/encoder/')
		self.decoder.save_weights('./model_checkpoints/decoder/')
		self.__last_saved = time.time()


	'''
		Load the last saved model checkpoint from the defined directory

	'''
	def load_checkpoint(self):
		self.encoder.load_weights('./model_checkpoints/encoder/')
		self.decoder.load_weights('./model_checkpoints/decoder/')



	'''
		Evaluate the model with labeled data
		(This is an interafce to self._evaluate_dataset)

		X:							(tensor) with input data
		y:							(tensor) with data labels (X paired)
		batch_size:					(int) representing desired batch size
		use_teacher_enforcement:	(boolean) use teacher enforcement
		verbose:					(boolean) print status

		returns:					self._evaluate_dataset() output
	'''
	def evaluate(self, X, y, batch_size=64, use_teacher_enforcement=False, verbose=True):

		# Create TF dataset
		ds, length = self.create_tf_dataset(X, y, batch_size)

		return self._evaluate_dataset(dataset=ds, total_data=length, batch_size=batch_size, use_teacher_enforcement=use_teacher_enforcement, verbose=verbose)



	'''
		Evaluate the model against an entire dataset and report loss

		dataset:					(tf.data.Dataset) dataset object with X, y labeled data pairs
		total_data:				 	(int) length of the dataset
		batch_size:				 	(int) representing desired batch size
		use_teacher_enforcement:	(boolean) use teacher enforment
		verbose:					(boolean) report progress and batch loss

		returns:					(float) mean dataset loss

	'''
	def _evaluate_dataset(self, dataset=None, total_data=0, batch_size=64, use_teacher_enforcement=False, verbose=True):
		# Check input
		if dataset is None:
			raise ValueError('dataset is required')

		if total_data < 1:
			raise ValueError('total_data is required')

		steps_per_epoch = total_data // batch_size
		total_loss = 0
		enc_hidden = self.encoder.initialize_hidden_state(batch_size)

		for (batch, (X, y)) in enumerate(dataset.take(steps_per_epoch)):
			pred, batch_loss = self.forward(X, y, use_teacher_enforcement=use_teacher_enforcement, training=False)
			total_loss += batch_loss.numpy()

			# Report batch
			if verbose:
				sys.stdout.write("\rTest batch: {}/{} - batch loss: {:.4f}".format(batch+1,
																				   steps_per_epoch,
																				   total_loss / (batch+1)))
				sys.stdout.flush()

		if verbose:
			sys.stdout.write("\n")
			sys.stdout.flush()

		return total_loss / steps_per_epoch


	'''
		Fixed loss function (SparseCategoricalCrossentropy)

	'''
	def loss_fn(self, real, pred):
		mask = tf.math.logical_not(tf.math.equal(real, 0))
		loss_ = self.loss_object(real, pred)
		mask = tf.cast(mask, dtype=loss_.dtype)
		loss_ *= mask
		return tf.reduce_mean(loss_)



	'''
		* Helper method, shold move somewhere else in the future

		Creates a tf.data.Dataset object to feed the network
		with tensors and in batches

		X:			  tensor with input data
		y:			  tensor with data labels (X paired)
		batch_size:	 int representing desired batch size

		returns:		(tf.data.Dataset) dataset object
						(int) length of the dataset

	'''
	def create_tf_dataset(self, X, y, batch_size=64):
		data_length = len(X)
		dataset = tf.data.Dataset.from_tensor_slices((X, y)).shuffle(data_length)
		dataset = dataset.batch(batch_size, drop_remainder=True)
		return dataset, data_length




class Evaluator():
	def __init__(self, model=None, dh=None, input_dict=None, output_dict=None):

		# Verify we got model and data handler injected during construction
		if model is None:
			print("Error: an EncoderDecoderWrapper instance must be provided")
			return False

		if dh is None:
			print("Error: a PreProcessor instance must be provided")
			return False

		if not isinstance(input_dict, tf.keras.preprocessing.text.Tokenizer):
			print("Error: invalid input_dict")
			return False

		if not isinstance(output_dict, tf.keras.preprocessing.text.Tokenizer):
			print("Error: invalid output_dict")
			return False

		self._model = model
		self.dh = dh
		self.input_dict = input_dict
		self.output_dict = output_dict


	def test_batch(self, X, y, batch_size=64, verbose=True):

		# Create TF dataset
		dataset, length = self._model.create_tf_dataset(X, y, batch_size)
		steps_per_batch = length // batch_size

		# Initialize
		total_loss = 0
		y_words = []
		yy_words = []

		# Get one batch (steps_per_epoch) and train from it
		for (batch, (X_batch, y_batch)) in enumerate(dataset.take(64)):
			batch_preds, batch_loss = self._model.test_step(X_batch, y_batch)

			# Accumulate loss
			total_loss += batch_loss.numpy()

			# Convert IDs to words and accumulate predictions
			yy_words.append(self.batch_to_words(batch_preds.numpy()))

			# Convert ground true ids to words
			y_words.append(self.batch_to_words(y_batch.numpy()))

		# Slice predicted sentences from <strt> to <stop>
		yy_words = [ yy[1: yy.index('<end>') if '<end>' in yy else len(yy)] for yy in yy_words ]
		y_words = [ y[1: y.index('<end>')] for y in y_words ]

		return yy_words, y_words, total_loss



	def test(self, sentence):
		attention_plot = np.zeros((self._model.output_dim, self._model.input_dim))

		sentence = self.dh.process_sentence(sentence)

		inputs = [self.input_dict.word_index[i] for i in sentence.split(' ')]
		inputs = tf.keras.preprocessing.sequence.pad_sequences([inputs],
															 maxlen=self._model.input_dim,
															 padding='post')
		inputs = tf.convert_to_tensor(inputs)

		result = ''

		hidden = [tf.zeros((1, self._model.encoder.units))]
		enc_out, enc_hidden = self._model.encoder(inputs, hidden)

		dec_hidden = enc_hidden
		dec_input = tf.expand_dims([self.output_dict.word_index['<start>']], 0)

		for t in range(self._model.output_dim):
			predictions, dec_hidden, attention_weights = self._model.decoder(dec_input,
																 dec_hidden,
																 enc_out)

			# storing the attention weights to plot later on
			attention_weights = tf.reshape(attention_weights, (-1, ))
			attention_plot[t] = attention_weights.numpy()

			predicted_id = tf.argmax(predictions[0]).numpy()

			result += self.output_dict.index_word[predicted_id] + ' '

			if self.output_dict.index_word[predicted_id] == '<end>':
				return result, sentence, attention_plot

			# the predicted ID is fed back into the model
			dec_input = tf.expand_dims([predicted_id], 0)

		return result, sentence, attention_plot

	def compute_ter_score(hyp, ref):
		return pyter.ter(hyp, ref)

	def evaluate_ter_batch(H, R):
		batch_score = 0
		for hyp, ref in zip(H, R):
			batch_score += compute_ter_score(hyp, ref)
		return batch_score / len(H)

	def batch_to_words(self, word_ids, dict=1):
		words = []
		for sentence in word_ids:
			s = []
			for wid in sentence:

				if wid < 1: # <start> or <stop>
					continue

				# Choose output dictionary
				if dict == 1:
					t = self.output_dict.index_word[wid]
				else:
					t = self.input_dict.index_word[wid]

				s.append(t)

			words.append(s)

		return words[0]






'''
	HELPERS

'''

def calc_normal_diff(a, b):
		max_err = max(a, b)
		error_diff = max_err - min(a, b)
		return error_diff * 100 / max_err
