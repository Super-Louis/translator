# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: train.py
# Python  : python3.6
# Time    : 18-9-28 14:25
from keras.layers import LSTM, Input, Dense, Bidirectional, Embedding, Dropout
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
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
    encode_input = pad_sequences(encode_input, maxlen=config['max_num'], truncating='post')
    decode_input = pad_sequences(decode_input, maxlen=config['max_num'], truncating='post')
    decode_output = pad_sequences(decode_output, maxlen=config['max_num'], truncating='post')
    decode_output = np.expand_dims(decode_output, 2)
    return encode_input, decode_input, decode_output

def train():
    # encoder
    encode_input, decode_input, decode_output = gen_datasets()
    encode_input, decode_input, decode_output = shuffle(encode_input, decode_input, decode_output)
    encoder_input = Input(shape=(None, ), name='encode_input')
    embedded_input = Embedding(config['en_voc_size'], 256, mask_zero=True, name="embedded_layer")(encoder_input)
    encoder = Bidirectional(LSTM(128, return_state=True), name="bi_lstm_layer", merge_mode="ave")
    encoder_outputs, state_h, state_c, state_h_rev, state_c_rev = encoder(embedded_input)
    encoder_states = [state_h, state_c, state_h_rev, state_c_rev]

    # decoder
    decoder_input = Input(shape=(None, ), name='decode_input')
    embedded_input2 = Embedding(config['ch_voc_size'], 256, mask_zero=True, name="embedded_layer2")(decoder_input)
    decoder = Bidirectional(LSTM(128, return_sequences=True, return_state=True), name="bi_lstm_layer2", merge_mode="ave")
    decoder_outputs, _, _, _, _ = decoder(embedded_input2, initial_state=encoder_states)
    decoder_outputs = Dropout(rate=0.5)(decoder_outputs)
    decoder_dense = Dense(config['ch_voc_size'], activation='softmax', name='dense_layer')
    decoder_outputs = decoder_dense(decoder_outputs)

    # train model
    model = Model([encoder_input, decoder_input], decoder_outputs)
    opt = optimizers.RMSprop(lr=0.01, decay=1e-6)
    model.compile(optimizer=opt, loss='sparse_categorical_crossentropy')
    model.summary()
    tb = TensorBoard(log_dir='./tb_logs/1009', histogram_freq=0, write_graph=True, write_images=False,
                     embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None)
    cp = ModelCheckpoint('./models/seq2seq.{epoch:02d}-{val_loss:.2f}.hdf5', monitor='val_loss', verbose=0,
                         save_best_only=False, save_weights_only=False, mode='auto', period=1)
    try:
        model.fit([encode_input, decode_input], decode_output, validation_split=0.2, callbacks=[tb, cp],
                  batch_size=config['batch_size'], epochs=config['epoch'])
    except KeyboardInterrupt:
        # Save model
        model.save('s2s_final')

if __name__ == '__main__':
    train()
