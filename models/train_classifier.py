import sys
import pandas as pd
from sqlalchemy import create_engine
import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer
nltk.download(['punkt', 'wordnet','averaged_perceptron_tagger','stopwords'])

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier 
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

import dill


def load_data(database_filepath):
    '''
    Load the data from the database and split it into X and Y
    Args:
        database_filepath (str): filepath of the sql database 
    Returns:
        X (numpy arr): feature vectors
        Y (numpy arr): multi-label vector output
        category_names (list of str): class names
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', engine)
    X = df['message'].values
    # 'child_alone' not included in category_names becuase it is all-zero and therefore INVALID for models such as SVM
    category_names = ['related', 'request', 'offer', 'aid_related', 'medical_help', \
                    'medical_products', 'search_and_rescue', 'security', 'military', \
                    'water', 'food', 'shelter', 'clothing', 'money', \
                    'missing_people', 'refugees', 'death', 'other_aid', \
                    'infrastructure_related', 'transport', 'buildings', 'electricity', \
                    'tools', 'hospitals', 'shops', 'aid_centers', 'other_infrastructure', \
                    'weather_related', 'floods', 'storm', 'fire', 'earthquake', 'cold', \
                    'other_weather', 'direct_report'] 
    Y = df[category_names].values
    return X, Y, category_names


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


def build_model():
    '''
    Build a multi-label classification model. 
    Optimize the hyperparameters with grid search and cross-validation
    Args: 
        None
    Returns:
        model (GridSearchCV): an ML model for multi-label classification
    '''
    mnb = MultinomialNB() # Best Parameters: alpha: 0.1, fit_prior: False, min_df: 0.0005
    lr = LogisticRegression() # Best Parameters: C: 1.0, class_weight: 'balanced', solver: 'lbfgs', min_df: 0.0005
    rf = RandomForestClassifier() # Best Parameters: class_weight: 'balanced', max_depth: None, min_samples_leaf: 4, n_estimators: 200, min_df: 0.001
    svm = SVC() # Best Parameters: C: 0.3, class_weight: 'balanced', kernel: 'rbf', min_df: 0.001
    evc = VotingClassifier(estimators=[
                                        ('mnb',mnb),
                                        ('lr',lr),
                                        ('rf',rf),
                                        ('svm',svm)
                                    ]) 
    pipeline = Pipeline([
                          ('vect', CountVectorizer(tokenizer = tokenize)),
                          ('tfidf', TfidfTransformer()),
                          ('clf', MultiOutputClassifier(rf))
                        ])
    
    parameters = {
        'vect__min_df': ([0.0005, 0.001, 0.005]), # vect

        #'clf__estimator__alpha': ([0.1, 0.5, 1.0]), # mnb
        #'clf__estimator__fit_prior': ([True, False]), # mnb

        #'clf__estimator__C': ([0.5, 1.0, 2.0]),# lr
        #'clf__estimator__solver':(['liblinear','lbfgs']),# lr
        #'clf__estimator__class_weight':(['balanced']), # lr

        'clf__estimator__n_estimators':([100, 200]), # rf
        'clf__estimator__max_depth':([50, None]), # rf
        'clf__estimator__min_samples_leaf':([1,4]), # rf
        'clf__estimator__class_weight':(['balanced']), # rf

        #'clf__estimator__C': ([0.1, 0.3, 0.5, 1.0]), # svm
        #'clf__estimator__kernel':(['linear','rbf']), # svm
        #'clf__estimator__class_weight':(['balanced']), # svm 

        #'vect__min_df': ([0.001]), # evc - vect
        #'clf__estimator__mnb__alpha': ([0.1]), # evc - mnb
        #'clf__estimator__mnb__fit_prior': ([False]), # evc - mnb
        #'clf__estimator__lr__C': ([1.0]),# evc - lr
        #'clf__estimator__lr__solver':(['liblinear']),# evc - lr
        #'clf__estimator__lr__class_weight':(['balanced']), # evc - lr 
        #'clf__estimator__rf__n_estimators':([200]), # evc - rf
        #'clf__estimator__rf__max_depth':([None]), # evc - rf
        #'clf__estimator__rf__min_samples_leaf':([4]), # evc - rf
        #'clf__estimator__rf__class_weight':(['balanced']), # evc - rf
        #'clf__estimator__svm__C': ([0.3]), # evc - svm
        #'clf__estimator__svm__kernel':(['rbf']), # evc - svm
        #'clf__estimator__svm__class_weight':(['balanced']), # evc - svm          
        }
    
    model = GridSearchCV(pipeline, 
                        param_grid=parameters, 
                        cv = 3, 
                        n_jobs = 1, 
                        verbose = 3, 
                        scoring = 'f1_macro'
                        )
    return model


def evaluate_model(model, X_train, Y_train, X_test, Y_test, category_names):
    '''
    Evaluate the multi-label classification model in terms of multi-label accuracy and f1-score
    Args: 
        model (sklearn): an ML model
        X_train (numpy arr): training vector
        Y_train (numpy arr): target vector relative to X_train
        X_test (numpy arr): testing vector
        Y_test (numpy arr): target vector relative to X_test
        category_names (list of str): class names
    Returns:
        None
    '''
    Y_pred = model.predict(X_test)
    Y_train_pred = model.predict(X_train)   
     
    print("\nBest Parameters:", model.best_params_)

    print("Train macro f1 score:", f1_score(Y_train, Y_train_pred, average = 'macro'))
    print("Test macro f1 score:", f1_score(Y_test, Y_pred, average = 'macro'))

    print("Test report:\n", classification_report(Y_test, Y_pred, target_names = category_names))


def save_model(model, category_names, tokenize, model_filepath):
    '''
    Wrap the trained model and the tokenizer function in a dictionary 
    and dump them together as a pickle file
    Args: 
        model (sklearn): an multi-label ML model
        category_names: labels of the model
        tokenizer (func): the tokenizer function
        model_filepath (str): path and pickle filename for saving the model
    Returns:
        None
    '''
    with open(model_filepath, 'wb') as out_strm: 
        dill.dump([model, category_names, tokenize], out_strm)


def main():
    '''
    Load the sql dataset; split the data into training and testing; build and train a model; 
    evaluate the model using various metrics; save the model as a pickle file
    '''
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
        evaluate_model(model, X_train, Y_train, X_test, Y_test, category_names)
        
        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, category_names, tokenize, model_filepath)
        print('Trained model saved!')

    else:
        print('''Please provide the filepath of the disaster messages database
              as the first argument and the filepath of the pickle file to
              save the model to as the second argument. \n
              Example: \n
              python train_classifier.py ../data/DisasterResponse.db classifier.pkl''')


if __name__ == '__main__':
    main()