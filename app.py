from flask import abort, request, Flask, jsonify, render_template
from services.thaigov_attention import AttentionInferencer
from services.thaigov_attention2 import AttentionInferencer2
from services.thaigov_attention_with_stopword import AttentionWithStopwordInferencer
from services.extractive import ExtractiveInferencer
from services.seq2seq import Seq2SeqInferencer

inferencers = {
    '1ext': ExtractiveInferencer(),
    'attn': AttentionInferencer(),
    'attn_stop': AttentionWithStopwordInferencer(),
    'seq2seq': Seq2SeqInferencer()
}

model_options = list(map(lambda model: (model, inferencers[model].get_name()) , inferencers))

app = Flask(__name__)

@app.route('/', methods=['GET'])
def hello():
    return render_template('index.html', options=model_options)

#! Just test not for fully used
@app.route('/preprocess', methods=['POST'])
def preprocess():
    if not request.json or not 'content' in request.json:
        abort(400)
    preprocessed_text = preprocess_text(request.json['content'])
    return jsonify({'preprocessed_text': preprocessed_text}), 201


# fully inference headline generation with word-based
@app.route('/inference/<inf_name>', methods=['POST'])
def infer(inf_name='thaigov'):
    if not request.json or not 'content' in request.json:
        abort(400)
    
    if inf_name not in inferencers:
        abort(404)

    content = request.json['content']
    
    inferred_headline = inferencers[inf_name].infer(content)

    return jsonify({'inferred_headline': inferred_headline}), 201


if __name__ == '__main__':
    app.run(debug=True, threaded=True, host='0.0.0.0')
