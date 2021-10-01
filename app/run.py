import json

from flask import Flask
from flask import render_template, request

import plotly

import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from data_wrangling import return_graphs, model, category_names, tokenize

app = Flask(__name__)


# index webpage that displays training dataset and receives user input text
@app.route('/')
@app.route('/index')
def index():
    graphs = return_graphs()
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i in range(len(graphs))]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_results = dict(zip(category_names, model.predict([query])[0]))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_results=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()