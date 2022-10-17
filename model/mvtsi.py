'''
This files includes the methods that define the attention model (MVTSI)
'''

import tensorflow as tf
from tensorflow.keras.layers import *


def scaled_dot_product_attention(q, k, v):
    '''
    Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    :param q: Query matrix
    :param k: Key matrix
    :param v: Value matrix
    :return: output, attention_weights
    '''
    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    '''This class builds the MultiHeadAttention layer as a keras layer'''

    def __init__(self, d_model, num_heads):
        '''
        :param d_model: Embedding dimension
        :param num_heads: Number of attention heads
        '''
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        '''
        Split the last dimension into (num_heads, depth).
        :param x: One of the matrices (q,k,v)
        :param batch_size: The Batch size
        :return: Transpose with shape (batch_size, num_heads, seq_len, depth)
        '''
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, q, k, v):
        '''
        :param q: Query matrix
        :param k: Key matrix
        :param v: Value matrix
        :return:
        '''
        batch_size = tf.shape(q)[0]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads,
        # depth)
        concat_attention = tf.reshape(scaled_attention,
                                      (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)
        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)
        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    '''
    Method defines the feed forward network for the last layer
    :param d_model: Embedding dimension
    :param dff: Dimension of feed forward network
    :return: Keras Sequential layer
    '''
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class EncoderLayer(tf.keras.layers.Layer):
    '''
    Wrapper class for the Encoder Layer
    '''

    def __init__(self, d_model, num_heads, dff, rate=0.1):
        '''
        :param d_model: Embedding dimension
        :param num_heads: Number of attention heads
        :param dff: Dimension of feed forward network
        :param rate: Learning rate
        '''
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = point_wise_feed_forward_network(d_model, dff)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)

    def call(self, q, k, v, training):
        '''
        :param q: Query matrix
        :param k: Key matrix
        :param v: Value matrix
        :param training: Boolean, whether model is training or not
        :return: output, attention_weights
        '''
        attn_output, attention_weights = self.mha(q, k, v)  # (batch_size, input_seq_len, d_model)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(attn_output + q)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn(out1)  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)
        return out2, attention_weights


class Encoder(tf.keras.layers.Layer):
    '''
    Wrapper class for the Encoder
    '''

    def __init__(self, num_layers, d_model, num_heads, dff, rate=0.1):
        '''
        :param num_layers: Number of encoder layers
        :param d_model: Embedding dimension
        :param num_heads: Number of attention heads
        :param dff: Dimension of feed forward network
        :param rate: learning rate
        '''
        super(Encoder, self).__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff, rate)
                           for _ in range(num_layers)]

    def call(self, q, k, v, training):
        '''
        :param q: Query matrix
        :param k: Key matrix
        :param v: Value matrix
        :param training: Boolean, whether model is training or not
        :return: q, attention_weights
        '''
        for i in range(self.num_layers):
            q, attention_weights = self.enc_layers[i](q, k, v, training)
        return q, attention_weights


class MVTSI(tf.keras.Model):
    '''
    Wrapper class for the final attention model
    '''

    def __init__(self, num_layers, d_model, num_heads, dff, nu_countries,
                 nu_years, nu_features, rate=0.1):
        '''
        :param num_layers: Number of encoder layers
        :param d_model: Embedding dimension
        :param num_heads: Number of attention heads
        :param dff: Dimension of feed forward network
        :param nu_countries: Number of countries used
        :param nu_years: Number of years used
        :param nu_features: Number of features used
        :param rate: Learning rate
        '''
        super(MVTSI, self).__init__()

        # Encoding Layers
        self.encoder = Encoder(num_layers, d_model, num_heads, dff, rate)
        # Embedding layers
        self.embedding_country = tf.keras.layers.Embedding(nu_countries.shape[0], 9)
        self.embedding_year = tf.keras.layers.Embedding(nu_years.shape[0], 4)
        self.embedding_feat = tf.keras.layers.Embedding(nu_features, 7)
        # Dense Layers
        self.first_layer = tf.keras.layers.Dense(d_model)
        self.final_layer = tf.keras.layers.Dense(1)

    def call(self, country, year, features_masked, miss_vals_masked,
             pos_enc, feature_nr, training):
        '''
        :param country: Information about country
        :param year: Information about year
        :param features_masked: Vector with masked features
        :param miss_vals_masked: Vector with masked missing values
        :param pos_enc: Positional encodings
        :param feature_nr: Vector of feature numbers
        :param training: Boolean, whether model is training or not
        :return: final_output, attention_weights
        '''
        # Embeddings
        country_emb = self.embedding_country(country)
        year_emb = self.embedding_year(year)
        feat_emb = self.embedding_feat(feature_nr)

        # concatenation (embeddings plus features)
        k = tf.concat([pos_enc,
                       tf.repeat(country_emb, pos_enc.shape[1], axis=1),
                       year_emb,
                       features_masked, miss_vals_masked, feat_emb], axis=2)
        k = self.first_layer(k)

        v = tf.concat([pos_enc,
                       tf.repeat(country_emb, pos_enc.shape[1], axis=1),
                       year_emb,
                       features_masked, miss_vals_masked, feat_emb], axis=2)
        v = self.first_layer(v)

        q = tf.concat([pos_enc,
                       tf.repeat(country_emb, pos_enc.shape[1], axis=1),
                       year_emb,
                       features_masked, miss_vals_masked, feat_emb], axis=2)
        q = self.first_layer(q)

        enc_output, attention_weights = self.encoder(q, k, v, training)  # (batch_size, inp_seq_len, d_model)

        final_output = self.final_layer(enc_output)  # (batch_size, tar_seq_len, target_vocab_size)

        return final_output, attention_weights


def loss_function(real, pred, miss_vals, mask):
    '''
    Custom MSE masked loss
    :param real: Real values
    :param pred: Predicted values
    :param miss_vals: Missing values
    :param mask: Masked values
    :return:
    '''
    real = real[mask==1]
    pred = pred[mask==1]
    missing = miss_vals[mask==1]
    real = tf.dtypes.cast(real, tf.float32)
    error = real-pred
    error = tf.square(error)
    #Exclude missing values
    loss = tf.math.sqrt(tf.math.reduce_mean(error[missing == 0]))
    return loss


def train_step(country, year, features, features_masked, miss_vals, miss_vals_masked,
               pos_enc, feature_nr, mask, optimizer):
    '''
    Perform a training step for the MVTSI model
    :param country: Information about country
    :param year: Information about year
    :param features: Feature vector
    :param features_masked: Vector with masked features
    :param miss_vals: Missing value vector
    :param miss_vals_masked: Vector with masked missing values
    :param pos_enc: Positional encodings
    :param feature_nr: Vector of feature numbers
    :param mask: Mask
    :param optimizer: Used optimizer
    :return: loss, pred, attention_weights
    '''
    training = True

    with tf.GradientTape() as tape:
        pred, attention_weights = MVTSI(country, year, features_masked, miss_vals_masked,
                                              pos_enc, feature_nr,  training)

        loss = loss_function(features, pred, miss_vals, mask)

    gradients = tape.gradient(loss, MVTSI.trainable_variables)
    optimizer.apply_gradients(zip(gradients, MVTSI.trainable_variables))
    return loss, pred, attention_weights