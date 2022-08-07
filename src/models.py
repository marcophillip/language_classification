from tensorflow.keras import layers
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import os,sys

print(sys.path)
from config import Config

config = Config()
max_length = config.max_length
count = config.count

def lstm_model():
    return tf.keras.models.Sequential([

        Embedding(count,128,input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16,dropout=0.2,return_sequences=True)),
        # tf.keras.layers.Conv1D(32,3,activation='relu'),
        tf.keras.layers.Bidirectional(LSTM(32,dropout=0.2,return_sequences=True)),
        GlobalAveragePooling1D(),
        Dense(6,activation='softmax')
    ])


def lstm_model2():
    return tf.keras.models.Sequential([
        tf.keras.layers.Input((5,count)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(16,dropout=0.2,return_sequences=True,input_shape=(5,count))),
        # tf.keras.layers.Conv1D(32,3,activation='relu'),
        tf.keras.layers.Bidirectional(LSTM(32,dropout=0.2,return_sequences=True)),
        GlobalAveragePooling1D(),
        Dense(6,activation='softmax')
    ])
   
'''
# source keras.io
'''
class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"),
             layers.Dense(ff_dim*2, activation="relu"),
             layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)





class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim,mask_zero=True)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions



def transformer():
    embed_dim =16  # Embedding sizor each token
    num_heads = 6 # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(max_length,))
    embedding_layer = TokenAndPositionEmbedding(max_length, count, embed_dim)
    x  = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
#     x = layers.Dropout(0.4)(x)
#     x = layers.Dense(32, activation="relu")(x)
#     x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(6, activation="softmax")(x)

    return tf.keras.Model(inputs=inputs, outputs=outputs)







