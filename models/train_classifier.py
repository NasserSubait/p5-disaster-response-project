import sys
import pandas as pd
import numpy as np
import sqlite3
import re
import pickle
import compress_pickle as cp

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download(['punkt', 'wordnet','stopwords'])

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score


def load_data(database_filepath):
    
    '''
    loading data from a database

    paramerter
    -------------
    database_filepath: file path of the database [str]
    

    return
    ------------
    X: a dataframe if the features
    Y: a dataframe with the target variables 
    category_names: a list of the columns read from the database

    ''' 
    
    conn = sqlite3.connect('../data/'+database_filepath)
    df = pd.read_sql('select * from messages',con = conn)
    X = df['message']
    Y = df.drop(['id','message','original','genre'],axis=1)
    category_names = Y.columns

    return X, Y, category_names


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


def build_model():
    
    '''
    building a model uses ML pipeline and uses grid search to optimize the model

    paramerter
    -------------
    None
    
    return
    ------------
    ML model ready to be used
   
    ''' 

    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [50, 100],
                  'clf__estimator__min_samples_split': [2, 3, 4],
                  'clf__estimator__criterion': ['entropy', 'gini']
                 }

    cv = GridSearchCV(pipeline, param_grid=parameters)

    
    return cv

def evaluate_model(model, X_test, Y_test, category_names):

    '''
    evaluate a ML model

    paramerter
    -------------
    model: a ML object model
    X_test: a series or dataframe
    Y_test: a series or dataframe
    category_names: a list of the categories that will be evaluated against
    
    return
    ------------
    a report of all categories in the databases and confusion matrix and f1 score
   
    ''' 
    
    y_predection = model.predict(X_test)

    for i in range(len(category_names)):
        print('category: '+category_names[i]+'\n')
        print( classification_report(Y_test.iloc[:, i].values, y_predection[:, i]))
        print('Accuracy of %25s: %.2f' %(category_names[i], accuracy_score(Y_test.iloc[:, i].values, y_predection[:,i])))
        print('\n----------------------------------------------------------------')
    


def save_model(model, model_filepath):

    """ Saves the model passed given the name of the file"""

    cp.dump(model,model_filepath,compression='lzma',set_default_extension=False)
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()