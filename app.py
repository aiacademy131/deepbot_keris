from flask import Flask
from flask import render_template
from train import intent

app = Flask(__name__)


@app.route('/chats/<text>', methods=['GET'])
def chats(text: str):
    sentiment = intent.getSentiment(text)
    print('sentiment', sentiment)
    return {
        'input': text,
        'intent': text,
        'entity': None,
        'state': 'SUCCESS',
        'answer': text,
        'sentiment': str(sentiment),
    }


# 챗봇 클라언트
@app.route('/')
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.template_folder = 'client'
    app.static_folder = 'client'
    app.run(port=3000, host='0.0.0.0', debug=True)