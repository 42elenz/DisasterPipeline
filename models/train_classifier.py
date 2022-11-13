# import basic libraries
import sys
import pandas as pd
import re

# import SQL-libraries
from sqlalchemy.engine import create_engine

# import nltk libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# import sk-learn libraries
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin

# download NLTK package
nltk.download(['stopwords', 'wordnet'])
nltk.download('averaged_perceptron_tagger')

#import pickle
import pickle

#import warning module
import warnings
warnings.filterwarnings('ignore')

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Class: Transformer that checks for every text entry in a Series if a sentence starts with a verb
    '''
    def starting_verb(self, text):
        '''
        Function: 
            checks for a text if one sentence starts with a verb
        Arg:
            text (string): text containing at least one sentence
        Return:
            Bool (bool): Whether one sentence starts with verb
        '''
        # splits text into lists of sentences
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
        # Check if sentence starts with verb
            if pos_tags:
                first_word, first_tag = pos_tags[0]
                if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                    return True
        return False

    def fit(self, X, y=None):
        '''
        Function: 
            returns itself.
        Arg:
            X (pd.Series): Series of text-messages
        Return:
            self (StartingVerbExtractor object): returns reference to object
        '''
        return self
    
    def transform(self, X):
        '''
        Function: 
            Applies starting_verb function on a Series.
        Arg:
            X (pd.Series): Series of text-messages.
        Return:
            X_tagged (pd.Dataframe): Booleans stating wheter a sentence in datapoint starts with verb.
        '''
        # Applies starting_verb function on series
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def load_data(database_filepath):
    '''
    Function: 
        loads data from database and returns X,Y and classnames.
    Args:
      database_filepath (str): database file name including path.
    Return:
      X (pd.DataFrame): messages to classify
      y (pd.DataFrame): labels for classification
      category_names (list): class names
    '''
    # load data from database
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    # splitting dataframe into X and Y
    X = df.iloc[:, :4]
    Y = df.iloc[:, 4:]
    return X["message"], Y, Y.columns

def tokenize(text):
    '''
    Function: 
        tokenizes given text string to list.
    Arg:
        text (string): twitter message.
    Return:
        cleaned (list): tokens of message.
    '''
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    # replace each url in text string with placeholder
    for url in detected_urls:
        text = text.replace(url, 'urlsub')
    # initiate Stemmer
    stemmer = PorterStemmer()
    # initiate Lemmatizer
    lemmatizer =  WordNetLemmatizer()
    # removing special characters and tranforming capital letters
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower()).split()
    # removing stop words
    text_tokens = [word for word in text if word not in stopwords.words('english')]
    # stemming tokens
    text_stems = [stemmer.stem(word) for word in text_tokens]
    # lemmatize tokens
    cleaned = [lemmatizer.lemmatize(word) for word in text_stems]
    return cleaned


def build_model():
    '''
    Function: 
        Builds classifier pipeline with the best hyperparameters and StartingVerbExtractor and returns it.
    Arg:
         N/A.
    Return:
        model (BaseEstimator): Instance of the built pipeline.
    '''
    # defining hyperparameter for the MultiOutputClassifier.
    params =  {'n_estimators': 200, 'min_samples_split': 4}
    # instanciating pipeline with customer tokenizer and StartingVerbExtractor.
    pipeline = Pipeline([
    ('features', FeatureUnion([
        ('vec', TfidfVectorizer(tokenizer = tokenize)),
        ('starting_verb', StartingVerbExtractor())
    ])),
    ('classifier', MultiOutputClassifier(RandomForestClassifier(**params)))
                        ])
    return(pipeline)

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function: 
        Prints Precision, Recall and F1 scores of the model for each output category of the trainingsset and modelaverages
        to stdout.
    Arg:
        model (BaseEstimator): trained model to be evaluated
        X_test (pd.Dataframe): predictor of test data
        Y_test (pd.Dataframe): labels of test data
        category_names (list): class names
    Return:
         N/A.
    '''
    # predict on the given trainingsset
    y_pred = model.predict(X_test)
    y_pred = pd.DataFrame(y_pred, columns=category_names)
    
    # Extract Precision, Recall and F1 scores for each output category
    scores = []
    for column in category_names:
        scores.append(classification_report(Y_test[column], y_pred[column]).split()[-4:-1])
    df_scores = pd.DataFrame(scores, index=category_names, columns = ['Precision', 'Recall', 'F1-Score'])
    df_scores = df_scores.astype(float)
    
    # calculate averages over all output categories
    averages = pd.Series([df_scores[column].mean() for column in df_scores.columns], index = df_scores.columns)
    
    # print category scores and averages
    print(df_scores)
    print(averages)

def save_model(model, model_filepath):
    '''
    Function: 
        saves model as pickle file.
    Args:
        cv: BaseEstimator. model to save
    Return:
      N/A
    '''
    pickle.dump(model, open(model_filepath, 'wb'))

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