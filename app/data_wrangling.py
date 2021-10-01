import pandas as pd

from plotly.graph_objs import Bar, Scatter

import dill
from sqlalchemy import create_engine

import re
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


from collections import Counter


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model and tokenize function
with open("../models/classifier.pkl", 'rb') as in_strm:
    model, category_names, tokenize = dill.load(in_strm)

# graph 1 - sample count for each category
category_distribution = df.iloc[:,4:].sum(axis=0).sort_values(ascending=False)
# graph 2 - sample count for each sample complexity (number of categories covered by each sample)
df['label_count'] = df.iloc[:,4:].sum(axis=1)
sample_complexity_distribution = df['label_count'].value_counts()
# graph 3 - the relation between token count and label count
df['token_count'] = df['message'].apply(lambda x:len(tokenize(x)))
# graph 4 - the top 10 popular tokens
all_tokens = []
for _, row in df.iterrows():
    all_tokens += tokenize(row['message'])
top_10_tokens_tuple = Counter(all_tokens).most_common(10)
top_10_tokens,top_10_tokens_count = list(zip(*top_10_tokens_tuple))


def return_graphs():
    # graph 1 - sample count for each category
    graph1 = dict(
                    data =  [
                        Bar(
                            x = list(category_distribution.index),
                            y = category_distribution.values
                        )
                    ],
                    layout = {
                        'title': 'Distribution of Sample Categories',
                        "titlefont": {"size": 18},
                        'yaxis': {
                            'title': "Sample Count"
                        },
                    }
                )

    # graph 2 - sample count for each sample complexity (number of categories covered by each sample)
    graph2 = {
                'data': [
                    Bar(
                        x = list(sample_complexity_distribution.index),
                        y = sample_complexity_distribution.values
                    )
                ],
                'layout': {
                    'title': 'Distribution of Sample Complexity (Number of Categories in Each Sample)',
                    'yaxis': {
                        'title': "Sample Count"
                    },
                }
            }

    # graph 3 - the relation between token count and label count
    graph3 = {
                'data': [
                    Scatter(
                        mode = 'markers',
                        x = df['token_count'],
                        y = df['label_count']
                    )
                ],
                'layout': {
                    'title': 'Token Count vs. Label Count',
                    'xaxis': {
                        'title': "Token Count"
                    },
                    'yaxis': {
                        'title': "Label Count"
                    },
                }
            }

    # graph 4 - the top 10 popular tokens
    graph4 = {
                'data': [
                    Bar(
                        x = top_10_tokens,
                        y = top_10_tokens_count
                    )
                ],
                'layout': {
                    'title': 'Top 10 Commonly Seen Tokens',
                    'yaxis': {
                        'title': "Token Count"
                    },
                }
            }

    graphs = [graph1, graph2, graph3, graph4]
    return graphs