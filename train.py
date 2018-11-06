# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: train.py
# Python  : python3.6
# Time    : 18-9-28 14:25
from keras.layers import LSTM, Input, Dense, Bidirectional, Embedding, \
    Dropout, Concatenate, RepeatVector, Activation, Dot, Lambda
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
import keras.backend as K
from gensim.models import word2vec
from sklearn.utils import shuffle
from keras import optimizers
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
import numpy as np
import json
with open('config.json', 'r') as f:
    config = json.load(f)

with open('data/ch_voc', 'r') as f:
    ch_voc = json.load(f)

with open('data/en_voc', 'r') as f:
    en_voc = json.load(f)

def gen_datasets():
    encode_input = np.load('data/encode_input.npy')
    decode_input = np.load('data/decode_input.npy')
    decode_output = np.load('data/decode_output.npy')
    # todo: pre/post padding ?
    encode_input = pad_sequences(encode_input, maxlen=config['max_num'], truncating='post', padding='post')
    decode_input = pad_sequences(decode_input, maxlen=config['max_num'], truncating='post', padding='post')
    decode_output = pad_sequences(decode_output, maxlen=config['max_num'], truncating='post', padding='post')
    decode_output = np.expand_dims(decode_output, 2)
    return encode_input, decode_input, decode_output

def get_word2vector():
    ch_w2v_matrix = np.zeros((len(ch_voc), 100)) # vector size 100
    ch_w2v = word2vec.Word2Vec.load('data/ch_word2vec.model').wv
    for w, i in ch_voc.items():
        try:
            ch_w2v_matrix[i] = ch_w2v.get_vector(w)
        except:
            pass
    en_w2v_matrix = np.zeros((len(en_voc), 100))  # vector size 100
    en_w2v = word2vec.Word2Vec.load('data/en_word2vec.model').wv
    for w, i in en_voc.items():
        try:
            en_w2v_matrix[i] = en_w2v.get_vector(w)
        except:
            pass
    return ch_w2v_matrix, en_w2v_matrix

ch_w2v_matrix, en_w2v_matrix = get_word2vector()

def simple_seq2seq():
    # encoder
    encoder_input = Input(shape=(None,), name='encode_input')
    embedded_input = Embedding(config['en_voc_size'], 100, weights=[en_w2v_matrix], trainable=False, mask_zero=True, name="embedded_layer")(encoder_input)
    encoder = LSTM(128, return_state=True, name="lstm_layer")
    encoder_outputs, state_h, state_c = encoder(embedded_input)
    encoder_states = [state_h, state_c]

    # decoder
    decoder_input = Input(shape=(None,), name='decode_input')
    embedded_input2 = Embedding(config['ch_voc_size'], 100, weights=[ch_w2v_matrix], trainable=False, mask_zero=True, name="embedded_layer2")(decoder_input)
    decoder = LSTM(128, return_sequences=True, return_state=True, name="lstm_layer2")
    decoder_outputs, _, _ = decoder(embedded_input2, initial_state=encoder_states)
    decoder_outputs = Dropout(rate=0.5)(decoder_outputs)
    decoder_dense = Dense(config['ch_voc_size'], activation='softmax', name='dense_layer')
    decoder_outputs = decoder_dense(decoder_outputs)

    model = Model([encoder_input, decoder_input], decoder_outputs)
    return model

def attention_seq2seq(input_len=config['max_num'], output_len=config['max_num']):
    # the decode input is not used
    encoder_input = Input(shape=(config['max_num'],), name='encode_input')
    embedded_input = Embedding(config['en_voc_size'], 100, weights=[en_w2v_matrix], trainable=False, name="embedded_layer")(encoder_input)
    encoder = LSTM(128, return_state=True, return_sequences=True, name="lstm_layer")
    encoder_outputs, state_h, state_c = encoder(embedded_input)
    decoder = LSTM(128, return_state=True, name="lstm_layer2")
    decoder_dense = Dense(config['ch_voc_size'], activation='softmax', name='dense_layer')
    # initial_input
    h = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], 128)))(encoder_input)
    c = Lambda(lambda X: K.zeros(shape=(K.shape(X)[0], 128)))(encoder_input)
    one_step_input = h
    initial_state = [h, c]
    outputs = []

    def softmax(x, axis=1):
        ndim = K.ndim(x)
        if ndim == 2:
            return K.softmax(x)
        elif ndim > 2:
            e = K.exp(x - K.max(x, axis=axis, keepdims=True))
            s = K.sum(e, axis=axis, keepdims=True)
            return e / s
        else:
            raise ValueError('Cannot apply softmax to a tensor that is 1D')

    def one_step_attention(encoder_outputs, one_step_input):
        # use mlp to get the weight
        one_step_input = RepeatVector(input_len)(one_step_input)
        concat = Concatenate(axis=-1)([encoder_outputs, one_step_input])
        e = Dense(10, activation="tanh")(concat)
        energies = Dense(1, activation="relu")(e)
        alphas = Activation(softmax)(energies)
        context = Dot(axes=1)([alphas, encoder_outputs])
        return context
    for i in range(output_len):
        context = one_step_attention(encoder_outputs, one_step_input)
        one_step_output, state_h, state_c = decoder(context, initial_state=initial_state)
        initial_state = [state_h, state_c]
        one_step_output = Dropout(rate=0.5)(one_step_output)
        one_step_input = one_step_output
        decoder_output = decoder_dense(one_step_output)
        outputs.append(decoder_output)
    model = Model(inputs=encoder_input, outputs=outputs)
    return model

def train(mode):
    model_dict = {
        "simple": simple_seq2seq,
        "attention": attention_seq2seq
    }
    encode_input, decode_input, decode_output = gen_datasets()
    encode_input, decode_input, decode_output = shuffle(encode_input, decode_input, decode_output)
    model = model_dict[mode]()
    opt = optimizers.RMSprop(lr=0.01, decay=1e-6)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    model.summary()
    tb = TensorBoard(log_dir='./tb_logs/1026', histogram_freq=0, write_graph=True, write_images=False,
                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    cp = ModelCheckpoint('./models/attention_seq2seq.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
                         save_best_only=False, save_weights_only=False, mode='auto', period=1)
    try:
        if mode == 'simple':
            input_ = [encode_input, decode_input]
            output_ = decode_output
        else:
            input_ = encode_input
            output_ = list(decode_output.swapaxes(0,1))
        model.fit(input_, output_, validation_split=0.2, callbacks=[tb, cp],
                  batch_size=config['batch_size'], epochs=10)
    except KeyboardInterrupt:
        # Save model
        model.save('s2s_final')
    else:
        model.save('s2s_final')

if __name__ == '__main__':
    train('attention')
