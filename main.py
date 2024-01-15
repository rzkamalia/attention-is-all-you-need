import tensorflow as tf

from config import ENGLISH_SEQUENCE_LENGTH, FRENCH_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM, NUM_LAYERS, NUM_HEADS, D_FF, WARM_UP_STEPS
from utils import Scheduler
from dataset import train_dataset, val_dataset, french_vectorize_layer, english_vectorize_layer
from model import Embeddings, TransformerEncoder, TransformerDecoder

encoder_inputs = tf.keras.layers.Input(shape = (None,), dtype = "int64", name = "input_1")
emb = Embeddings(ENGLISH_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)
x = emb(encoder_inputs)
enc_mask = emb.compute_mask(encoder_inputs)

for _ in range(NUM_LAYERS):
    x = TransformerEncoder(EMBEDDING_DIM, D_FF, NUM_HEADS)(x)
encoder_outputs = x

decoder_inputs = tf.keras.layers.Input(shape = (None,), dtype = "int64", name = "input_2")

x = Embeddings(FRENCH_SEQUENCE_LENGTH, VOCAB_SIZE, EMBEDDING_DIM)(decoder_inputs)
for i in range(NUM_LAYERS):
      x = TransformerDecoder(EMBEDDING_DIM, D_FF, NUM_HEADS)(x, encoder_outputs, enc_mask)
x = tf.keras.layers.Dropout(0.5)(x)
decoder_outputs = tf.keras.layers.Dense(VOCAB_SIZE, activation = "softmax")(x)

transformer = tf.keras.Model(
    [encoder_inputs, decoder_inputs], decoder_outputs, name = "transformer"
)
# transformer.summary()

lr_scheduled = Scheduler(EMBEDDING_DIM, WARM_UP_STEPS)

transformer.compile(
    loss = tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer = tf.keras.optimizers.Adam(lr_scheduled))

# train
def train():
    history = transformer.fit(train_dataset, validation_data = val_dataset, epochs = 5)
    transformer.save_weights('transformers.h5')

# train()

# inference
index_to_word = {x: y for x, y in zip(range(len(french_vectorize_layer.get_vocabulary())), french_vectorize_layer.get_vocabulary())}
def translator(english_sentence):
    transformer.load_weights('transformers.h5')

    tokenized_english_sentence = english_vectorize_layer([english_sentence])
    shifted_target = 'starttoken'

    for i in range(FRENCH_SEQUENCE_LENGTH):
        tokenized_shifted_target = french_vectorize_layer([shifted_target])
        output = transformer.predict([tokenized_english_sentence, tokenized_shifted_target])
        french_word_index = tf.argmax(output, axis = -1)[0][i].numpy()
        current_word = index_to_word[french_word_index]
        if current_word == 'endtoken':
            break
        shifted_target += ' ' + current_word
    return shifted_target[11:]

result = translator('What makes you think that it is not true?')
print(result)