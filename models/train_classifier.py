import sys
import pandas as pd
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize, RegexpTokenizer
from nltk.corpus import stopwords
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, f1_score


def load_data(database_filepath):
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message'].values
    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', \
                    'medical_products', 'search_and_rescue', 'security', 'military', \
                    'child_alone', 'water', 'food', 'shelter', 'clothing', 'money', \
                    'missing_people', 'refugees', 'death', 'other_aid', \
                    'infrastructure_related', 'transport', 'buildings', 'electricity', \
                    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', \
                    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', \
                    'other_weather', 'direct_report']
    Y = df[category_names].values
    return X, Y, category_names


def tokenize(text):
    # normalize, split, and remove punctuations
    words = RegexpTokenizer(r'\w+').tokenize(text.lower())
     
    # filter out stopwords
    stopwords_set = set(stopwords.words("english"))
    words_stop = [word for word in words if word not in stopwords_set]
     
    return words_stop


def tokenize_w_lem(text):
    # normalize, split, and remove punctuations
    words = RegexpTokenizer(r'\w+').tokenize(text.lower())
    
    # part-of-speech tagging
    words_pos = pos_tag(words)
    
    # filter out stopwords
    stopwords_set = set(stopwords.words("english"))
    words_stop = [(word,p) for word,p in words_pos if word not in stopwords_set]
    
    # lemmatize verb and noun
    lemmatizer = WordNetLemmatizer()
    words_lem = [(lemmatizer.lemmatize(word, pos = 'v' if p.startswith('VB') else 'n'), p) for word, p in words_stop]
    
    words_final = [word+'_'+p for word,p in words_lem] 
    return words_final


def build_model():
    '''
    pipeline = Pipeline([
                        ('feat_union', FeatureUnion([
                            ('text_pipeline', Pipeline([
                                ('vect', CountVectorizer()),
                                ('tfidf', TfidfTransformer())
                            ]))
                            #,('starting_verb', StartingVerbExtractor())
                        ])),
                        ('clf', MultiOutputClassifier(RandomForestClassifier()))
                    ])
    parameters = {
        'feat_union__vect__tokenizer': (tokenize, tokenize_w_lem)        
    }
    '''
    
    pipeline = Pipeline([('vect', CountVectorizer()),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultiOutputClassifier(RandomForestClassifier()))])
    
    parameters = {
        'vect__tokenizer': (tokenize, tokenize_w_lem)        
    }
    
    model = GridSearchCV(pipeline, 
                      param_grid=parameters, 
                      cv = 3, 
                      #n_jobs = 2, 
                      verbose = 3, 
                      scoring = 'accuracy')
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    Y_pred = model.predict(X_test)
    accuracy = (Y_pred == Y_test).mean()
    f1 = f1_score(Y_test, Y_pred, average = 'micro')    
    print("\nBest Parameters:", model.best_params_)
    print("Accuracy:", accuracy)
    print("F1 score:", f1)
    print(classification_report(Y_test, Y_pred))


def save_model(model, model_filepath):
    model_filepath = pickle.dumps(model)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        #save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('''Please provide the filepath of the disaster messages database
              as the first argument and the filepath of the pickle file to
              save the model to as the second argument. \n
              Example: \n
              python train_classifier.py ../data/DisasterResponse.db classifier.pkl''')


if __name__ == '__main__':
    main()