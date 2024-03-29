dataset = "fra-eng/fra.txt"

VOCAB_SIZE = 20000
ENGLISH_SEQUENCE_LENGTH = 32
FRENCH_SEQUENCE_LENGTH = 32
EMBEDDING_DIM = 512
BATCH_SIZE = 128

NUM_BATCHES = int(200000 / BATCH_SIZE)

D_FF = 2048
NUM_HEADS = 8
NUM_LAYERS = 1
NUM_EPOCHS = 10

WARM_UP_STEPS = 4000