from services.inferencer import Inferencer
from config.hparams import create_hparams
from tensorflow.keras.models import load_model
from tensorflow.keras.activations import softmax
from tensorflow.keras.preprocessing.sequence import pad_sequences
from preprocessed.preprocessed_thaigov import word_to_index, preprocess_text
import pickle
import numpy as np

hparams = create_hparams()
maxlen = hparams['maxlen']

MODEL_ATTN_PATH = './model/v6_remove_stopword_50_20_attention_best.h5'
# MODEL_ATTN_PATH = './model/200_20_v2.2_normal_best_200_20_attention_best.h5'

with open('./decode/dictionary/dict_t_50_20_v2.pkl', 'rb') as f:
    dict_t = pickle.load(f)

with open('./decode/dictionary/rev_dict_t_50_20_v2.pkl', 'rb') as f:
    rev_dict_t = pickle.load(f)

class AttentionInferencer(Inferencer):

    def __init__(self):
        super().__init__()

        def softMaxAxis1(x):
            return softmax(x,axis=1)
            
        self.model = load_model(MODEL_ATTN_PATH, custom_objects={ 'softmax': softmax })

    def get_name(self):
        return 'Attention LSTM w/ teacher forcing'

    def preprocess(self, content):
        preprocessed_text = preprocess_text(content)
        index_seq = word_to_index(preprocessed_text, dict_t)
        index_seq = pad_sequences([index_seq], maxlen=maxlen, padding='post')
        return index_seq

    def postprocess(self, pred):
        out = []
        for token in pred:
            out.append(rev_dict_t[np.argmax(token)])
        return ' '.join(out).replace('for_keras_zero_padding', '')

    def infer(self, content):
        index_seq = self.preprocess(content)
        start_input = np.array([dict_t["<s>"]] * 1)
        pred = self.model.predict([np.array(index_seq), start_input])
        return self.postprocess(pred)
