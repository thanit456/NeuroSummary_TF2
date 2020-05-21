from services.inferencer import Inferencer
from preprocessed.preprocessed_thaigov import preprocess_text, word_to_index
from tensorflow.keras.preprocessing.sequence import pad_sequences
from config.hparams import create_hparams
import pickle
import tensorflow as tf
import os
import numpy as np

hparams = create_hparams()
maxlen_input = hparams['maxlen']
maxlen_output = hparams['maxlen_output']

with open('./decode/dictionary/gru_dict.pkl', 'rb') as f:
    dict_t = pickle.load(f)

with open('./decode/dictionary/gru_rev_dict.pkl', 'rb') as f:
    rev_dict_t = pickle.load(f)

#define_n_first_words
N_FIRST_CONTENT = maxlen_input #50
N_FIRST_HEADLINE = maxlen_output #22

BATCH_SIZE = 64
embedding_dim = 256
units = 1024
vocab_inp_size = len(dict_t)
vocab_tar_size = len(dict_t)

## ! use only n first words for headline generation
def use_first_n_words(df_content, n, start_stop):
    new_ls = []
    for content in df_content:
        if start_stop:
            tmp = content[:n-2]
            tmp.insert(0, '<s>')
            tmp.append('</s>')
        else:
            tmp = content[:n]
    new_ls.append(tmp)
    return new_ls

class Encoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, enc_units, batch_sz):
    super(Encoder, self).__init__()
    self.batch_sz = batch_sz
    self.enc_units = enc_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.enc_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')

  def call(self, x, hidden):
    x = self.embedding(x)
    output, state = self.gru(x, initial_state = hidden)
    return output, state

  def initialize_hidden_state(self):
    return tf.zeros((self.batch_sz, self.enc_units))

class BahdanauAttention(tf.keras.layers.Layer):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, query, values):
    # query hidden state shape == (batch_size, hidden size)
    # query_with_time_axis shape == (batch_size, 1, hidden size)
    # values shape == (batch_size, max_len, hidden size)
    # we are doing this to broadcast addition along the time axis to calculate the score
    query_with_time_axis = tf.expand_dims(query, 1)

    # score shape == (batch_size, max_length, 1)
    # we get 1 at the last axis because we are applying score to self.V
    # the shape of the tensor before applying self.V is (batch_size, max_length, units)
    score = self.V(tf.nn.tanh(
        self.W1(query_with_time_axis) + self.W2(values)))

    # attention_weights shape == (batch_size, max_length, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * values
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class Decoder(tf.keras.Model):
  def __init__(self, vocab_size, embedding_dim, dec_units, batch_sz):
    super(Decoder, self).__init__()
    self.batch_sz = batch_sz
    self.dec_units = dec_units
    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.dec_units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc = tf.keras.layers.Dense(vocab_size)

    # used for additive attention
    self.attention = BahdanauAttention(self.dec_units)
    
    # used for multiplicative attention
    # self.attention = LuongAttention(self.dec_units)

  def call(self, x, hidden, enc_output):
    # enc_output shape == (batch_size, max_length, hidden_size)
    context_vector, attention_weights = self.attention(hidden, enc_output)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # output shape == (batch_size * 1, hidden_size)
    output = tf.reshape(output, (-1, output.shape[2]))

    # output shape == (batch_size, vocab)
    x = self.fc(output)

    return x, state, attention_weights

def evaluate(sentence):
    attention_plot = np.zeros((maxlen_output, maxlen_input))

    inputs = sentence.reshape(1, maxlen_input)

    inputs = tf.convert_to_tensor(inputs)

    result = '<s> '

    hidden = [tf.zeros((1, units))]
    enc_out, enc_hidden = encoder(inputs, hidden)

    dec_hidden = enc_hidden
    dec_input = tf.expand_dims([dict_t['<s>']], 0)

    for t in range(maxlen_output):
        predictions, dec_hidden, attention_weights = decoder(dec_input, dec_hidden, enc_out)

        # storing the attention weights to plot later on
        attention_weights = tf.reshape(attention_weights, (-1, ))
        attention_plot[t] = attention_weights.numpy()

        predicted_id = tf.argmax(predictions[0]).numpy()

        result += rev_dict_t[predicted_id] + ' '

        if rev_dict_t[predicted_id] == '</s>':
            return result, sentence, attention_plot

        # the predicted ID is fed back into the model
        dec_input = tf.expand_dims([predicted_id], 0)

    return result, sentence, attention_plot

def index2sentence(indexes):
    result = ''
    for i, index in enumerate(indexes):
        result += rev_dict_t[index] + ' '

    if rev_dict_t[index] == '</s>':
        return result
    
    return result

def translate(sentence):
    result, sentence, attention_plot = evaluate(np.array(sentence))
    return result

optimizer = tf.keras.optimizers.Adam()
encoder = Encoder(vocab_inp_size, embedding_dim, units, BATCH_SIZE)
decoder = Decoder(vocab_tar_size, embedding_dim, units, BATCH_SIZE)

checkpoint_dir = './model/gru_additive'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
checkpoint.restore(os.path.join(checkpoint_dir, 'ckpt-10'))
    
class Seq2SeqInferencer(Inferencer):

    def __init__(self):
        super().__init__()

    def get_name(self):
        return 'GRU Additive'

    def preprocess(self, content):
        tokenized_text = preprocess_text(content)
        content = use_first_n_words(tokenized_text, n=N_FIRST_CONTENT, start_stop=False)
        index_seq = word_to_index(content, dict_t)
        index_seq = pad_sequences([index_seq], maxlen=maxlen_input, padding='post')
        return index_seq

    def postprocess(self, text):
        return text.replace('<s>', '').replace('</s>', '').strip()

    def infer(self, content):
        preprocessed_text = self.preprocess(content)
        pred = translate(preprocessed_text)
        return self.postprocess(pred)