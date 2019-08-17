import json
import plotly
import pandas as pd


import nltk
from nltk.tokenize import word_tokenize
#nltk.download(['punkt', 'wordnet','stopwords', 'averaged_perceptron_tagger'])
import re
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):

    '''
    A toknize function that process a text to be ready for ML pipeline 
    that follows NLP pipeline

    paramerter
    -------------
    text: a string of text [str]
    

    return
    ------------
    clear_text: a tokenize text
   
    ''' 

    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    clear_text = [PorterStemmer().stem(w) for w in lemmed]
    
    return clear_text

# load data
engine = create_engine('sqlite:///../data/DisasterDB.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifer_c.lzma")

#model = joblib.load("../models/your_model_name.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()