import tensorflow as tf 

from config import dataset, ENGLISH_SEQUENCE_LENGTH, FRENCH_SEQUENCE_LENGTH, VOCAB_SIZE, BATCH_SIZE, NUM_BATCHES
from utils import selector, separator

text_dataset = tf.data.TextLineDataset(dataset)

english_vectorize_layer = tf.keras.layers.TextVectorization(
    standardize = 'lower_and_strip_punctuation',
    max_tokens = VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length = ENGLISH_SEQUENCE_LENGTH
)

french_vectorize_layer = tf.keras.layers.TextVectorization(
    standardize = 'lower_and_strip_punctuation',
    max_tokens = VOCAB_SIZE,
    output_mode = 'int',
    output_sequence_length = FRENCH_SEQUENCE_LENGTH
)

split_dataset = text_dataset.map(selector)

init_dataset = text_dataset.map(separator)

english_training_data = init_dataset.map(lambda x, y: x) # input x,y and output x
english_vectorize_layer.adapt(english_training_data) # adapt the vectorize_layer to the training data

french_training_data=init_dataset.map(lambda x, y: y) # input x,y and output y
french_vectorize_layer.adapt(french_training_data) # adapt the vectorize_layer to the training data

def vectorizer(inputs, output):
  return {'input_1': english_vectorize_layer(inputs['input_1']),
          'input_2': french_vectorize_layer(inputs['input_2'])}, french_vectorize_layer(output)

dataset = split_dataset.map(vectorizer)

dataset = dataset.shuffle(2048).unbatch().batch(BATCH_SIZE).prefetch(buffer_size = tf.data.AUTOTUNE)

train_dataset = dataset.take(int(0.9 * NUM_BATCHES))
val_dataset = dataset.skip(int(0.9 * NUM_BATCHES))