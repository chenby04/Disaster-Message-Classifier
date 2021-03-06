import pandas as pd
import dill
from plotly.graph_objs import Bar, Scatter
from sqlalchemy import create_engine
from collections import Counter

import re
import nltk
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer


# load cleaned data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('messages', engine)

# load model and tokenize function
with open("../models/classifier.pkl", 'rb') as in_strm:
    model, category_names, tokenize = dill.load(in_strm)

# prepare data for plotting
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
    '''
    Generate graphs using Plotly
    '''
    # graph 1 - sample count for each category
    graph1 = {
                'data': [
                    Bar(
                        x = list(category_distribution.index),
                        y = category_distribution.values
                    )
                ],
                'layout': {
                    'yaxis': {
                        'title': "Message Count"
                    },
                    'margin':{'l':100, 'r':100, 't':0, 'b':120}
                }
            }

    # graph 2 - sample count for each sample complexity (number of categories covered by each sample)
    graph2 = {
                'data': [
                    Bar(
                        x = list(sample_complexity_distribution.index),
                        y = sample_complexity_distribution.values
                    )
                ],
                'layout': {
                    'xaxis': {
                        'title': "Message Complexity (# of Classes per Message)"
                    },
                    'yaxis': {
                        'title': "Message Count"
                    },
                    'margin':{'l':100, 'r':100, 't':0, 'b':50}
                }
            }

    # graph 3 - the relation between token count and label count
    graph3 = {
                'data': [
                    Scatter(
                        mode = 'markers',
                        x = df['token_count'],
                        y = df['label_count'],
                    )
                ],
                'layout': {
                    'xaxis': {
                        'title': "Token Count",
                        #'type': "log"
                    },
                    'yaxis': {
                        'title': "Class Count",
                        #'type': "log"
                    },
                    
                    'margin':{'l':200, 'r':200, 't':0, 'b':50}
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
                    # 'title': 'Top 10 Commonly Seen Tokens',
                    'xaxis': {
                        'title': "Token"
                    },
                    'yaxis': {
                        'title': "Token Count"
                    },
                    'bargap': 0.3,
                    'margin':{'l':200, 'r':200, 't':0, 'b':50}
                }
            }

    graphs = [graph1, graph2, graph3, graph4]
    return graphs