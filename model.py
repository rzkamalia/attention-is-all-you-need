import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Layer, Embedding, Dense, LayerNormalization

def positional_encoding(model_size, sequence_length):
    output = []
    for pos in range(sequence_length):
        PE = np.zeros((model_size))
        for i in range(model_size):
            if i%2 == 0:
                PE[i] = np.sin(pos / (10000**(i / model_size)))
            else:
                PE[i] = np.cos(pos / (10000**((i-1) / model_size)))            
        output.append(tf.expand_dims(PE, axis = 0))
    out = tf.concat(output, axis = 0)
    out = tf.expand_dims(out, axis = 0)
    return tf.cast(out, dtype = tf.float32)

class Embeddings(Layer):
  def __init__(self, sequence_length, vocab_size, embed_dim):
    super(Embeddings, self).__init__()
    self.token_embeddings = Embedding(input_dim = vocab_size, output_dim = embed_dim)
    self.sequence_length = sequence_length
    self.vocab_size = vocab_size
    self.embed_dim = embed_dim

  def call(self, inputs):
    embedded_tokens = self.token_embeddings(inputs)
    embedded_positions = positional_encoding(self.embed_dim, self.sequence_length)
    return embedded_tokens + embedded_positions

  def compute_mask(self, inputs, mask = None):
    return tf.math.not_equal(inputs, 0)

class CustomSelfAttention(Layer):
    def __init__(self, model_size):
        super(CustomSelfAttention, self).__init__()
        self.model_size = model_size
        
    def call(self, query, key, value, masking):
        # compute scores
        score = tf.matmul(query, key, transpose_b = True)
        # scaling
        score /= tf.math.sqrt(tf.cast(self.model_size, tf.float32))
        # masking
        masking = tf.cast(masking, dtype = tf.float32)
        score += (1.-masking) * -1e10
        # attention_weights
        attention = tf.nn.softmax(score, axis = -1) * masking
        # output
        head = tf.matmul(attention, value)
        return head

class CustomMultiHeadAttention(Layer):
    def __init__(self, num_heads, key_dim):
        super(CustomMultiHeadAttention, self).__init__()

        self.num_heads = num_heads
        self.dense_q = [Dense(key_dim // num_heads) for _ in range(num_heads)]
        self.dense_k = [Dense(key_dim // num_heads) for _ in range(num_heads)]
        self.dense_v = [Dense(key_dim // num_heads) for _ in range(num_heads)]
        self.dense_o = Dense(key_dim)
        self.self_attention = CustomSelfAttention(key_dim)

    def call(self, query, key, value, attention_mask):
        heads = []
        for i in range(self.num_heads):
            head = self.self_attention(self.dense_q[i](query), self.dense_k[i](key), self.dense_v[i](value), attention_mask)
            heads.append(head)
        heads = tf.concat(heads, axis = 2)
        heads = self.dense_o(heads)
        return heads
    
class TransformerEncoder(Layer):
    def __init__(self, embed_dim, dense_dim, num_heads):
        super(TransformerEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = CustomMultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [
                Dense(dense_dim, activation = "relu"),
                Dense(embed_dim)
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, mask = None):
        if mask is not None:
            mask = tf.cast(mask[:, tf.newaxis, :], dtype = "int32")
            T = tf.shape(mask)[2]
            padding_mask = tf.repeat(mask, T, axis = 1)

        attention_output = self.attention(query = inputs, key = inputs, value = inputs, attention_mask = padding_mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

class TransformerDecoder(Layer):
    def __init__(self, embed_dim, latent_dim, num_heads):
        super(TransformerDecoder, self).__init__()
        self.embed_dim = embed_dim
        self.latent_dim = latent_dim
        self.num_heads = num_heads
        self.attention_1 = CustomMultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.attention_2 = CustomMultiHeadAttention(num_heads = num_heads, key_dim = embed_dim)
        self.dense_proj = tf.keras.Sequential(
            [
                Dense(latent_dim, activation = "relu"),
                Dense(embed_dim)
            ]
        )
        self.layernorm_1 = LayerNormalization()
        self.layernorm_2 = LayerNormalization()
        self.layernorm_3 = LayerNormalization()
        self.supports_masking = True

    def call(self, inputs, encoder_outputs, enc_mask, mask = None):
        if mask is not None:
            causal_mask = tf.linalg.band_part(tf.ones([tf.shape(inputs)[0], tf.shape(inputs)[1], tf.shape(inputs)[1]], dtype = tf.int32), -1, 0)
            mask = tf.cast(mask[:, tf.newaxis, :], dtype = "int32")
            enc_mask = tf.cast(enc_mask[:, tf.newaxis, :], dtype = "int32")

            T = tf.shape(mask)[2]
            padding_mask = tf.repeat(mask, T, axis = 1)
            cross_attn_mask = tf.repeat(enc_mask, T , axis = 1)
            combined_mask = tf.minimum(padding_mask, causal_mask)

        attention_output_1 = self.attention_1(query = inputs, key = inputs, value = inputs, attention_mask = combined_mask)
        out_1 = self.layernorm_1(inputs + attention_output_1)
        attention_output_2 = self.attention_2(query = out_1, key = encoder_outputs, value = encoder_outputs, attention_mask = cross_attn_mask)
        out_2 = self.layernorm_2(out_1 + attention_output_2)
        proj_output = self.dense_proj(out_2)
        return self.layernorm_3(out_2 + proj_output)

# VOCAB_SIZE = 10
# EMBEDDING_DIM = 15
# DENSE_DIM = 20

# test_input = tf.constant([[2, 4, 9, 5, 0]])
# sequence_length = 5

# emb = Embeddings(sequence_length, VOCAB_SIZE, EMBEDDING_DIM)
# emb_out = emb(test_input)

# mask = emb.compute_mask(test_input)
# padding_mask = tf.cast(tf.repeat(mask, repeats = tf.shape(mask)[1], axis = 0), dtype = tf.int32)
# encoder_outputs = TransformerEncoder(EMBEDDING_DIM, DENSE_DIM, sequence_length)(emb_out, padding_mask)

# enc_mask = mask
# decoder_outputs = TransformerDecoder(EMBEDDING_DIM, DENSE_DIM, sequence_length)(emb_out, encoder_outputs, enc_mask)