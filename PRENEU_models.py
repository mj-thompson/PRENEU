import sys
from sklearn import metrics
import gc
import tensorflow as tf
# tf.config.experimental.set_visible_devices([], 'GPU')
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras import Input
from tensorflow.keras.layers import (
    GRU,
    Activation,
    ActivityRegularization,
    Add,
    BatchNormalization,
    Bidirectional,
    Concatenate,
    Conv1D,
    Dense,
    Dropout,
    Flatten,
    Lambda,
    Layer,
    LayerNormalization,
    #MultiHeadAttention
)
import os
from tensorflow.keras.models import Model
import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
from sklearn import metrics
from tensorflow.keras.layers import Input, Conv1D, LSTM, Bidirectional, Dense, Flatten, Dropout, BatchNormalization, Masking, Concatenate, Multiply, GRU
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from math import floor

## Need to get all the models built. Our choices are
# CAN, enabling howevery many attention heads we want, should plot the pos embeddings as well
# LSTM- just final states, also hidden states
# CNN stacked and smol
# CNN with dilated and residual connections
# CNN with dilated connections/residual solo?

# Code thanks to interpretable splicing model from Oded Regev lab
from sklearn import metrics
def binary_KL(y_true, y_pred):
    # return K.mean(K.binary_crossentropy(y_pred, y_true)-K.binary_crossentropy(y_true, y_true), axis=-1)   # this is for the Ubuntu machine in Courant
    return tf.keras.backend.mean(
        tf.keras.backend.binary_crossentropy(y_true, y_pred)
        - tf.keras.backend.binary_crossentropy(y_true, y_true),
        axis=-1,
    )  # this i


# early stopping callback
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
                                            patience=10, 
                                            verbose=1, 
                                            mode='min', 
                                            restore_best_weights=True)
# reduce learning rate callback
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                factor=0.2,
                                                patience=6, 
                                                min_lr=1e-7,
                                                mode='min',
                                                verbose=1) 
def get_layer_output(model, index, X):
    temp = tf.keras.Model(inputs=model.inputs, outputs=model.layers[index].output)
    return temp.predict(X)



def get_cnn(
    input_length=150,    
    dense_number=64,
    num_filters=120,
    num_filters2=30,
    num_filters3=30,
    num_filters4=60,
    filter_width=7,
    filter_width2=3,
    filter_width3=5,
    filter_width4=2,
    dropout_rate=0.4,
    num_heads=10,
    key_len=20,
    l2_regularization=1e-4,
    pool_size=5,
    batch_size=100,
    act_reg=0,
    indim=4,
    outact="linear"
):

    ########################
    ## Define model logic ##
    ########################
    
    # Inputs
    seq_input = Input(shape=(input_length, indim),name="seq_input")
    
    # Sequence conv 1
    primary_conv=Conv1D(filters=num_filters, kernel_size=filter_width, name="seq_conv",
                       padding="same",use_bias=False)(seq_input)
    out_seq_conv=Activation("exponential",name="seq_conv_activation")(primary_conv)
    # Pooling
    out_seq_conv=keras.layers.MaxPool1D(pool_size=floor(filter_width/2),padding="same",name="seq_conv_pool")(out_seq_conv)
    
    # Sequence conv 2
    out_seq_conv=Conv1D(filters=num_filters2, kernel_size=filter_width2, name="seq_conv2",
                       padding="same",use_bias=False)(out_seq_conv)
    out_seq_conv=Activation("relu",name="seq_conv_activation2")(out_seq_conv)
    # Pooling
    out_seq_conv=keras.layers.MaxPool1D(pool_size=floor(filter_width2/2),padding="same",name="seq_conv_pool2")(out_seq_conv)

    
    # Sequence conv 3
    out_seq_conv=Conv1D(filters=num_filters3, kernel_size=filter_width3, name="seq_conv3",
                       padding="same",use_bias=False)(out_seq_conv)
    out_seq_conv=Activation("relu",name="seq_conv_activation3")(out_seq_conv)
    # Pooling
    out_seq_conv=keras.layers.MaxPool1D(pool_size=floor(filter_width3/2),padding="same",name="seq_conv_pool3")(out_seq_conv)

    
    # Sequence conv 4
    out_seq_conv=Conv1D(filters=num_filters4, kernel_size=filter_width4, name="seq_conv4",
                       padding="same",use_bias=False)(out_seq_conv)
    out_seq_conv=Activation("relu",name="seq_conv_activation4")(out_seq_conv)
    # Pooling
    out_seq_conv=keras.layers.MaxPool1D(pool_size=floor(filter_width4/2),padding="same",name="seq_conv_pool4")(out_seq_conv)

    
    out_seq=Flatten(name="Flatten")(out_seq_conv)
    
    

    ### low-rank weight matrices can lead to redundant filters
    ### https://people.cs.umass.edu/~arunirc/downloads/pubs/redundant_filter_dltp2017.pdf
    ### if this is true, test with and without?
    dense=Dense(floor(dense_number*1.5),kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense1")(out_seq)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    
    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense2")(dense)
    dense=BatchNormalization(name="BN2")(dense)
    dense=Activation("relu",name="dense_activation2")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout2")(dense)
    
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    # model.compile(optimizer="adam", loss="bce")
    
    return model





class MultiHeadAttention_wpos(keras.layers.Layer):
    def __init__(self, d_model, num_heads, embedding_size=None,name="MHA"):
        super(MultiHeadAttention_wpos, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.embedding_size = d_model if embedding_size == None else embedding_size

        assert d_model % self.num_heads == 0 and d_model % 6 == 0

        self.depth = d_model // self.num_heads

        self.wq = keras.layers.Dense(d_model, use_bias=False)
        self.wk = keras.layers.Dense(d_model, use_bias=False)
        self.wv = keras.layers.Dense(d_model, use_bias=False)

        self.r_k_layer = keras.layers.Dense(d_model, use_bias=False)
        self.r_w = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True,
                              name="r_w")
        self.r_r = tf.Variable(tf.random_normal_initializer(0, 0.5)(shape=[1, self.num_heads, 1, self.depth]), trainable=True,
                              name="r_r")

        self.dense = keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size, seq_len):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, seq_len, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]
        seq_len = tf.constant(q.shape[1])

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size, seq_len)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size, seq_len)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size, seq_len)  # (batch_size, num_heads, seq_len_v, depth)
        q = q / tf.math.sqrt(tf.cast(self.depth, dtype=tf.float32))

        pos = tf.range(-seq_len + 1, seq_len, dtype=tf.float32)[tf.newaxis]
        feature_size=self.embedding_size//6

        seq_length = tf.cast(seq_len, dtype=tf.float32)
        exp1 = f_exponential(tf.abs(pos), feature_size, seq_length=seq_length)
        exp2 = tf.multiply(exp1, tf.sign(pos)[..., tf.newaxis])
        cm1 = f_central_mask(tf.abs(pos), feature_size, seq_length=seq_length)
        cm2 = tf.multiply(cm1, tf.sign(pos)[..., tf.newaxis])
        gam1 = f_gamma(tf.abs(pos), feature_size, seq_length=seq_length)
        gam2 = tf.multiply(gam1, tf.sign(pos)[..., tf.newaxis])

        # [1, 2seq_len - 1, embedding_size]
        positional_encodings = tf.concat([exp1, exp2, cm1, cm2, gam1, gam2], axis=-1)
        positional_encodings = keras.layers.Dropout(0.1)(positional_encodings)

        # [1, 2seq_len - 1, d_model]
        r_k = self.r_k_layer(positional_encodings)

        # [1, 2seq_len - 1, num_heads, depth]
        r_k = tf.reshape(r_k, [r_k.shape[0], r_k.shape[1], self.num_heads, self.depth])
        r_k = tf.transpose(r_k, perm=[0, 2, 1, 3])
        # [1, num_heads, 2seq_len - 1, depth]

        # [batch_size, num_heads, seq_len, seq_len]
        content_logits = tf.matmul(q + self.r_w, k, transpose_b=True)

        # [batch_size, num_heads, seq_len, 2seq_len - 1]
        relative_logits = tf.matmul(q + self.r_r, r_k, transpose_b=True)
        # [batch_size, num_heads, seq_len, seq_len]
        relative_logits = relative_shift(relative_logits)

        # [batch_size, num_heads, seq_len, seq_len]
        logits = content_logits + relative_logits
        #### add in masking capability ####
        # logits = tf.where(tf.equal(mask, 1), -10000.0, logits)
        attention_map = tf.nn.softmax(logits)

        # [batch_size, num_heads, seq_len, depth]
        attended_values = tf.matmul(attention_map, v)
        # [batch_size, seq_len, num_heads, depth]
        attended_values = tf.transpose(attended_values, perm=[0, 2, 1, 3])

        concat_attention = tf.reshape(attended_values, [batch_size, seq_len, self.d_model])

        output = self.dense(concat_attention)

        return output, attention_map



#------------------------------------------------------------------------------------------
# Positional encoding functions for Multi-Head Attention
#------------------------------------------------------------------------------------------

## Code thanks to tfomics and papers from Chandana Rajesh and Peter Koo's github.
def f_exponential(positions, feature_size, seq_length=None, min_half_life=3.0):
    if seq_length is None:
        seq_length = tf.cast(tf.reduce_max(tf.abs(positions)) + 1, dtype=tf.float32)
    max_range = tf.math.log(seq_length) / tf.math.log(2.0)
    half_life = tf.pow(2.0, tf.linspace(min_half_life, max_range, feature_size))
    half_life = tf.reshape(half_life, shape=[1]*positions.shape.rank + half_life.shape)
    positions = tf.abs(positions)
    outputs = tf.exp(-tf.math.log(2.0) / half_life * positions[..., tf.newaxis])
    return outputs

def f_central_mask(positions, feature_size, seq_length=None):
    center_widths = tf.pow(2.0, tf.range(1, feature_size + 1, dtype=tf.float32)) - 1
    center_widths = tf.reshape(center_widths, shape=[1]*positions.shape.rank + center_widths.shape)
    outputs = tf.cast(center_widths > tf.abs(positions)[..., tf.newaxis], tf.float32)
    return outputs

def f_gamma(positions, feature_size, seq_length=None):
    if seq_length is None:
        seq_length = tf.reduce_max(tf.abs(positions)) + 1
    stdv = seq_length / (2*feature_size)
    start_mean = seq_length / feature_size
    mean = tf.linspace(start_mean, seq_length, num=feature_size)
    mean = tf.reshape(mean, shape=[1]*positions.shape.rank + mean.shape)
    concentration = (mean / stdv) ** 2
    rate = mean / stdv**2
    def gamma_pdf(x, conc, rt):
        log_unnormalized_prob = tf.math.xlogy(concentration - 1., x) - rate * x
        log_normalization = (tf.math.lgamma(concentration) - concentration * tf.math.log(rate))
        return tf.exp(log_unnormalized_prob - log_normalization)
    probabilities = gamma_pdf(tf.abs(tf.cast(positions, dtype=tf.float32))[..., tf.newaxis], concentration, rate)
    outputs = probabilities / tf.reduce_max(probabilities)
    return outputs

def relative_shift(x):
    to_pad = tf.zeros_like(x[..., :1])
    x = tf.concat([to_pad, x], -1)
    _, num_heads, t1, t2 = x.shape
    x = tf.reshape(x, [-1, num_heads, t2, t1])
    x = tf.slice(x, [0, 0, 1, 0], [-1, -1, -1, -1])
    x = tf.reshape(x, [-1, num_heads, t1, t2 - 1])
    x = tf.slice(x, [0, 0, 0, 0], [-1, -1, -1, (t2 + 1) // 2])
    return x


class getattnmask(tf.keras.layers.Layer):
  def __init__(self, num_repeats,name="attn_mask"):
    super(getattnmask, self).__init__()
    self.num_repeats = num_repeats

  def call(self, inputs):
    sequence_masks_attn=tf.squeeze(inputs)
    sequence_masks_attn=tf.abs(tf.subtract(inputs,tf.constant(1,dtype=tf.float32)))
    sequence_masks_attn=tf.repeat(sequence_masks_attn,repeats=sequence_masks_attn.shape[-2],
             axis=-1)
    sequence_masks_attn=tf.transpose(sequence_masks_attn,[0,2,1])
    sequence_masks_attn=sequence_masks_attn[:,tf.newaxis,:,:]
    return sequence_masks_attn

def get_can(
    input_length=150,    
    dense_number=64,
    num_filters=50,
    filter_width=5,
    dropout_rate=0.4,
    num_heads=10,
    key_len=20,
    l2_regularization=1e-4,
    pool_size=5,
    batch_size=100,
    act_reg=0,
    indim=4,
    outact="linear"
):

    ########################
    ## Define model logic ##
    ########################
    
    # Inputs
    seq_input = Input(shape=(input_length, indim),name="seq_input")
    
    # Sequence conv
    primary_conv=Conv1D(filters=num_filters, kernel_size=filter_width, name="seq_conv",
                       padding="same",use_bias=False)(seq_input)
    out_seq_conv=Activation("exponential",name="seq_conv_activation")(primary_conv)
    
    # Pooling?
    if isinstance(pool_size,int):
        out_seq_conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(out_seq_conv)
    
    # Dropout of convolution
    out_seq_conv=Dropout(.1,name="seq_conv_act_dropoout")(out_seq_conv)
    
    # Pass in the attention with the masks
    out_seq_attn, wseq=MultiHeadAttention_wpos(num_heads=num_heads,
                                                d_model=num_heads*key_len,name="MHA")(
        out_seq_conv,out_seq_conv,out_seq_conv)
    
    # Dropout of attention
    out_seq_attn=Dropout(.1,name="seq_attn_dropout")(out_seq_attn)
    out_seq_attn=Flatten(name="Flatten")(out_seq_attn)
    

    ### low-rank weight matrices can lead to redundant filters
    ### https://people.cs.umass.edu/~arunirc/downloads/pubs/redundant_filter_dltp2017.pdf
    ### if this is true, test with and without?
    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(out_seq_attn)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    # model.compile(optimizer="adam", loss="bce")
    
    return model




def get_lstm_seq_hidden_states(input_length=251,
    exonlength=63,
    dense_number=24,
    lstm_feats=24,
    num_filters=60,
    filter_width=7,pool_size="no",l2_regularization=1e-4,
    regularization=1e-3,outact="sigmoid",
                          dropout_rate=0.4):
    # Define input shape
    input_shape = (input_length, 4)  # 182 positions, 4 features per position
    inputs = Input(shape=input_shape)
    
    # Get convs on all
    conv=Conv1D(filters=int(num_filters), 
                     kernel_size=filter_width, 
                     use_bias=False, padding='same',name="conv_joint")(inputs)
    conv=Activation("exponential",name="seq_conv_activation")(conv)
        # Pooling?
    if isinstance(pool_size,int):
        conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(conv)
    
    # Dropout of convolution
    conv=Dropout(.1,name="seq_conv_act_dropoout")(conv)
    
    
    # 2. BiLSTM for Long-Range Dependencies
    x = Bidirectional(LSTM(lstm_feats, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(conv)
    x = Bidirectional(LSTM(lstm_feats, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(x)
    x = Flatten()(x)

    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(x)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=inputs, outputs=out)
    
    auroc = tf.keras.metrics.AUC(curve='ROC', name='auroc')
    aupr = tf.keras.metrics.AUC(curve='PR', name='aupr')
    model.compile(optimizer="adam", loss="mse")


    # Model Summary
    model.summary()
    return model




def get_lstm_final_cells(input_length=251,
    exonlength=63,
    dense_number=24,
    lstm_feats=24,
    num_filters=60,
    dropout_rate=0.4,
    filter_width=7,pool_size="no",l2_regularization=1e-4,
    regularization=1e-3,outact="linear"):
    # Define input shape
    input_shape = (input_length, 4)  # 182 positions, 4 features per position
    inputs = Input(shape=input_shape)
    
    # Get convs on all
    conv=Conv1D(filters=int(num_filters), 
                     kernel_size=filter_width, 
                     use_bias=False, padding='same',name="conv_joint")(inputs)
    conv=Activation("exponential",name="seq_conv_activation")(conv)
        # Pooling?
    if isinstance(pool_size,int):
        conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(conv)
    
    # Dropout of convolution
    conv=Dropout(.1,name="seq_conv_act_dropoout")(conv)
    
    x = Bidirectional(LSTM(lstm_feats, return_sequences=True, return_state=True,dropout=0.2, recurrent_dropout=0.2))(conv)
    x = Bidirectional(LSTM(lstm_feats, return_sequences=False, return_state=False,dropout=0.2, recurrent_dropout=0.2))(x)
    

    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(x)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer="adam", loss="mse")


    # Model Summary
    model.summary()
    return model

def get_lstm_seq_hidden_states_and_finalcell(input_length=251,
    exonlength=63,
    dense_number=24,
    lstm_feats=24,
    num_filters=60,
    dropout_rate=0.4,
    filter_width=7,pool_size="no",l2_regularization=1e-4,
    regularization=1e-3,outact="linear"):
    # Define input shape
    input_shape = (input_length, 4)  # 182 positions, 4 features per position
    inputs = Input(shape=input_shape)
    
    # Get convs on all
    conv=Conv1D(filters=int(num_filters), 
                     kernel_size=filter_width, 
                     use_bias=False, padding='same',name="conv_joint")(inputs)
    conv=Activation("exponential",name="seq_conv_activation")(conv)
        # Pooling?
    if isinstance(pool_size,int):
        conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(conv)
    
    # Dropout of convolution
    conv=Dropout(.1,name="seq_conv_act_dropoout")(conv)
    
    x = Bidirectional(LSTM(lstm_feats, return_sequences=True, return_state=True,dropout=0.2, recurrent_dropout=0.2))(conv)
    all_h_states,forward_h, forward_c, backward_h, backward_c = Bidirectional(LSTM(lstm_feats, return_sequences=True, return_state=True,dropout=0.2, recurrent_dropout=0.2))(x)
    
    all_h_states=Flatten()(all_h_states)
    
    
    x=Concatenate()([all_h_states, forward_c,backward_c])   

    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(x)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer="adam", loss="mse")


    # Model Summary
    model.summary()
    return model





from tensorflow.keras import layers, Model



def dilated_residual_block(x, filters, kernel_size, dilation_rate, dropout_rate=0.0):
    """
    A residual block with dilated convolutions.
    x: input tensor of shape (batch, time, channels)
    filters: number of filters for the convolutions
    kernel_size: size of the convolution kernel
    dilation_rate: dilation rate for the convolutions
    dropout_rate: dropout rate (if > 0, applies dropout after first conv)
    Returns a tensor of the same shape as x.
    """
    shortcut = x
    # First dilated conv with ReLU.
    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    # Second dilated conv without activation.
    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    # If necessary, match shortcut channels.
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv1D(filters, kernel_size=1, padding="same")(shortcut)
        shortcut = layers.BatchNormalization()(shortcut)

    # Add shortcut and apply ReLU.
    x = layers.Add()([x, shortcut])
    return layers.Activation("relu")(x)








def dilated_block_no_resid(x, filters, kernel_size, dilation_rate, dropout_rate=0.0):
    """
    A residual block with dilated convolutions.
    x: input tensor of shape (batch, time, channels)
    filters: number of filters for the convolutions
    kernel_size: size of the convolution kernel
    dilation_rate: dilation rate for the convolutions
    dropout_rate: dropout rate (if > 0, applies dropout after first conv)
    Returns a tensor of the same shape as x.
    """
    # First dilated conv with ReLU.
    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)

    # Second dilated conv without activation.
    x = layers.Conv1D(filters, kernel_size, dilation_rate=dilation_rate,
                      padding="same")(x)
    x = layers.BatchNormalization()(x)
    
    return layers.Activation("relu")(x)


def get_conv_baseline(
    input_length=200,    
    dense_number=48,
    num_filters=100,
    filter_width=7,
    dropout_rate=0.4,
    l2_regularization=1e-4,
    batch_size=100,
    indim=4,
    outact="linear",
    pool_size="no"
):

    ########################
    ## Define model logic ##
    ########################
    
    # Inputs
    seq_input = Input(shape=(input_length, indim),name="seq_input")
    
    # Sequence conv
    primary_conv=Conv1D(filters=num_filters, kernel_size=filter_width, name="seq_conv",
                       padding="same",use_bias=False)(seq_input)
    out_seq_conv=Activation("exponential",name="seq_conv_activation")(primary_conv)
    
    # Pooling?
    if isinstance(pool_size,int):
        out_seq_conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(out_seq_conv)
    
    # Dropout of convolution
    out_seq_conv=Dropout(.1,name="seq_conv_act_dropoout")(out_seq_conv)

    

    ### low-rank weight matrices can lead to redundant filters
    ### https://people.cs.umass.edu/~arunirc/downloads/pubs/redundant_filter_dltp2017.pdf
    ### if this is true, test with and without?
    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(out_seq_conv)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    
    dense=Flatten(name="Flatten")(dense)
    
    dense=Dense(int(dense_number/2),kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha2")(dense)
    dense=BatchNormalization(name="BN2")(dense)
    dense=Activation("relu",name="dense_activation2")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout2")(dense)
    
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=seq_input, outputs=out)
    model.compile(optimizer="adam", loss="mse")
    # model.compile(optimizer="adam", loss="bce")
    
    return model

def build_dil_conv(seq_len=498, input_dim=4,
    dense_number=128,
    lstm_feats=64,
    num_blocks=5,
    num_filters=100,
    filter_width=7,
                   dropout_rate=0.1,
                   initial_filters=80,
    regularization=1e-3,
                     pool_size="no",
    l2_regularization=1e-4):
    kernel_size=filter_width
    inputs = layers.Input(shape=(seq_len, input_dim), name="input_sequence")
    
        # Get convs on all
    conv=Conv1D(filters=int(num_filters), 
                     kernel_size=filter_width, 
                     use_bias=False, padding='same',name="conv_joint")(inputs)
    conv=Activation("exponential",name="seq_conv_activation")(conv)
        # Pooling?
    if isinstance(pool_size,int):
        conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(conv)
    
    # Dropout of convolution
    conv=Dropout(.1,name="seq_conv_act_dropoout")(conv)
    
    
    
    x = conv
    
    for i in range(num_blocks):
        dilation_rate = 3 ** i  # 1, 2, 4, 8, ...
        x = dilated_residual_block(x, filters=floor(initial_filters/(i+1)), kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, dropout_rate=dropout_rate)
    x = Flatten()(x)
    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(x)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer="adam", loss="mse")
    return model

def build_dil_conv_noresid(seq_len=498, input_dim=4,
    dense_number=128,
    lstm_feats=64,
    num_blocks=5,
    num_filters=100,
    filter_width=7,
                   dropout_rate=0.1,
                   initial_filters=80,
    regularization=1e-3,
                     pool_size="no",
    l2_regularization=1e-4):
    kernel_size=filter_width
    inputs = layers.Input(shape=(seq_len, input_dim), name="input_sequence")
    
        # Get convs on all
    conv=Conv1D(filters=int(num_filters), 
                     kernel_size=filter_width, 
                     use_bias=False, padding='same',name="conv_joint")(inputs)
    conv=Activation("exponential",name="seq_conv_activation")(conv)
        # Pooling?
    if isinstance(pool_size,int):
        conv=keras.layers.MaxPool1D(pool_size=pool_size,name="seq_conv_pool")(conv)
    
    # Dropout of convolution
    conv=Dropout(.1,name="seq_conv_act_dropoout")(conv)
    
    
    
    x = conv
    
    for i in range(num_blocks):
        dilation_rate = 3 ** i  # 1, 2, 4, 8, ...
        x = dilated_block_no_resid(x, filters=floor(initial_filters/(i+1)), kernel_size=kernel_size,
                                   dilation_rate=dilation_rate, dropout_rate=dropout_rate)
    x = Flatten()(x)
    dense=Dense(dense_number,kernel_regularizer=keras.regularizers.L1L2(l2_regularization,l2_regularization),
               name="Dense_mha")(x)
    dense=BatchNormalization(name="BN")(dense)
    dense=Activation("relu",name="dense_activation")(dense)
    dense=Dropout(dropout_rate,name="dense_dropout")(dense)
    out=Dense(1,activation="linear",name="output_activation")(dense)
    # create model
    model = Model(inputs=inputs, outputs=out)
    
    model.compile(optimizer="adam", loss="mse")
    return model




def get_model(modtype="can",inputshape=200,
    dense_number=64,
    num_filters=100,
    filter_width=7,
    batch_size=100,
    epochs=100,
    regularization=0.00,
    pool_size=3,
    keylen=6,j=0,
    dropout_rate=.2,
    h2="0.9",
    N="100000",
    p="10",
    alpha="0.3",
    outact="linear",
    outactname="linear",
    savesuffix="test"):
    if modtype=="cnn":
        model=get_cnn(input_length=inputshape,
                        dense_number=dense_number,num_filters=num_filters,
                        filter_width=filter_width,pool_size=pool_size,
                        key_len=keylen,num_heads=8,
              dropout_rate=dropout_rate,
                        l2_regularization=regularization,
                 outact=outact)
    elif modtype=="can":
        model=get_can(input_length=inputshape,
                    dense_number=dense_number,num_filters=num_filters,
                    filter_width=filter_width,pool_size=pool_size,
                    key_len=keylen,num_heads=8,
                    l2_regularization=regularization,
                     outact=outact)
    elif modtype=="can_nopool":
        dense_number=48
        model=get_can(input_length=inputshape,
                    dense_number=dense_number,num_filters=num_filters,
                    filter_width=filter_width,pool_size="no",
                    key_len=keylen,num_heads=4,
                    l2_regularization=regularization,
                     outact=outact)
    elif modtype=="lstm_seq":
        dense_number=24
        model=get_lstm_seq_hidden_states(input_length=inputshape,
                    dense_number=dense_number,num_filters=num_filters,
                    filter_width=filter_width,
                     outact="linear",lstm_feats=24)
    elif modtype=="lstm_final":
        model=get_lstm_final_cells(input_length=inputshape,
                    dense_number=dense_number*2,num_filters=num_filters,
                    filter_width=filter_width,
                     outact="linear",lstm_feats=64)
    elif modtype=="lstm_seq_and_final":
        dense_number=24
        model=get_lstm_seq_hidden_states_and_finalcell(input_length=inputshape,
                    dense_number=dense_number,num_filters=num_filters,
                    filter_width=filter_width,
                     outact="linear",lstm_feats=24)
    elif modtype=="dilcnn_resid_f":
        model=build_dil_conv_f(seq_len=inputshape,
                dense_number=dense_number/4,num_filters=num_filters,
                filter_width=filter_width)
    elif modtype=="dilcnn_noresid_f":
        model=build_dil_conv_noresid_f(seq_len=inputshape,
                dense_number=dense_number/4,num_filters=num_filters,
                filter_width=filter_width)
    elif modtype=="baseline":
        model=get_conv_baseline(input_length=inputshape, filter_width=filter_width)
    else:
        dense_number=64
        num_dense=4
        model=get_dense(input_length=inputshape,
                    dense_number=dense_number,num_filters=num_filters,
                    filter_width=filter_width,pool_size=pool_size,
                    key_len=keylen,num_heads=8,
                    l2_regularization=regularization,
                     outact=outact,num_dense=num_dense)
    model_name="/u/project/halperin/mjthomps/motif_sims/fit_models/fit_"+modtype+savesuffix
    model_name+="_"+str(dense_number)+"_"+str(num_filters)+"_"
    model_name+=str(filter_width)+"_"+str(batch_size)+"_"+str(epochs)
    model_name+="_"+str(regularization) + "pool_" + str(pool_size)
    model_name+="_" + str(keylen) + "_es_" + str(6) + "_DO_" + str(dropout_rate)
    model_name+="_outact_"+outactname
    results_path="./"
    print(model_name)
    model_dir = os.path.join(model_name+'_weights.h5')
    print(model.summary())
    return model, model_dir
