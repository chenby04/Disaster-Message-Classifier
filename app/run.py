import json
import plotly
import pandas as pd

from flask import Flask
from flask import render_template, request, jsonify

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])

from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    '''
    Tokenize the text by removing puctuation, normalizing case, splitting by space, removing stop words, 
    lemmatizing, and stemming
    Args:
        text (str): a message 
    Returns:
        tokens (list of str): tokens
    '''
    # replace punctuations with space and normalize all letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize 
    tokens = word_tokenize(text)  
    
    # lemmatize, stem and remove stop words
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(lemmatizer.lemmatize(word)) for word in tokens if word not in stop_words]
    return tokens

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load("../models/classifier.pk")


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
    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', \
                    'medical_products', 'search_and_rescue', 'security', 'military', \
                    'water', 'food', 'shelter', 'clothing', 'money', \
                    'missing_people', 'refugees', 'death', 'other_aid', \
                    'infrastructure_related', 'transport', 'buildings', 'electricity', \
                    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', \
                    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', \
                    'other_weather', 'direct_report'] 
    classification_results = dict(zip(category_names, model.predict([query])[0]))

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