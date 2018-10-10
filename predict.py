# -*- coding: utf-8 -*-
# Author  : Super~Super
# FileName: predict.py
# Python  : python3.6
# Time    : 18-10-9 10:35
import json
import nltk
import jieba
import numpy as np
from keras import Model
from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model
from keras.layers import Input

with open('config.json', 'r') as f:
    config = json.load(f)

with open('data/ch_voc', 'r') as f:
    ch_voc = json.load(f)

with open('data/en_voc', 'r') as f:
    en_voc = json.load(f)

ch_voc_rev = {v:k for k, v in ch_voc.items()}
en_voc_rev = {v:k for k, v in en_voc.items()}

def load_s2s_model():
    # form encoder_model
    model = load_model('s2s_final')
    encoder_inputs = model.input[0]
    encoder_outputs, state_h, state_c, state_h_rev, state_c_rev = model.layers[4].output
    encoder_states = [state_h, state_c, state_h_rev, state_c_rev]
    encoder_model = Model(encoder_inputs, encoder_states)

    # form decoder_model
    decoder_inputs = model.input[1]
    decoder_embedding = model.layers[3]
    embedded_input = decoder_embedding(decoder_inputs)
    decoder_state_input_h = Input(shape=(128,), name='input_3')
    decoder_state_input_c = Input(shape=(128,), name='input_4')
    decoder_state_input_h_rev = Input(shape=(128,), name='input_3')
    decoder_state_input_c_rev = Input(shape=(128,), name='input_4')
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c,
                             decoder_state_input_h_rev, decoder_state_input_c_rev]
    decoder_lstm = model.layers[5]
    decoder_outputs, state_h_dec, state_c_dec, state_h_dec_rev, state_c_dec_rev = decoder_lstm(
        embedded_input, initial_state=decoder_states_inputs)
    decoder_states = [state_h_dec, state_c_dec, state_h_dec_rev, state_c_dec_rev]
    dropout_layer = model.layers[6]
    decoder_outputs = dropout_layer(decoder_outputs)
    dense_layer = model.layers[7]
    decoder_outputs = dense_layer(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)
    return encoder_model, decoder_model

def predict_trans(inputs):
    encoder_model, decoder_model = load_s2s_model()
    seq = inputs.strip().lower().replace(' - ', '-')
    words = nltk.word_tokenize(seq, )
    seqs = [en_voc.get(w, en_voc["<unk>"]) for w in words]
    sentences = pad_sequences([seqs], maxlen=config['max_num'], truncating='post')
    states = encoder_model.predict(sentences)
    stop_condition = False
    target_sentence = [[ch_voc[config['start_word']]]]
    predict_words = []
    while not stop_condition:
        output_tokens, h, c, h_rev, c_rev = decoder_model.predict(
            [target_sentence] + states)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = ch_voc_rev[sampled_token_index]
        predict_words.append(sampled_char)

        if (sampled_char == ch_voc[config['end_word']] or
           len(predict_words) > config['max_num']):
            stop_condition = True

        # Update the target sequence (of length 1).
        target_sentence = [[sampled_token_index]]

        # Update states
        states = [h, c, h_rev, c_rev]
    return predict_words

if __name__ == '__main__':
    predict_trans('I love you.')
