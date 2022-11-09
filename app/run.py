# import basic libraries
import sys
import pandas as pd
import re
import json

# import SQL-libraries
from sqlalchemy.engine import create_engine

# import nltk libraries
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

# import sk-learn libraries
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

# download NLTK package
nltk.download(['stopwords', 'wordnet'])
nltk.download('averaged_perceptron_tagger')

# import pickle
import pickle

# import warning module
import warnings
warnings.filterwarnings('ignore')

# import flask package
from flask import Flask
from flask import render_template, request, jsonify

# import plotly package
import plotly
from plotly.graph_objs import Bar

app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    '''
    Class: Transformer that checks for every text entry in a Series if a sentence starts with a verb
    '''
    def starting_verb(self, text):
        '''
        Function: 
            checks for a text if one sentence starts with a verb.
        Arg:
            text (string): text containing at least one sentence.
        Return:
            Bool (bool): Whether one sentence starts with verb.
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
            X (pd.Series): Series of text-messages.
        Return:
            self (StartingVerbExtractor object): returns reference to object.
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


# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data for genres
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # extract data for categories
    categories = df.iloc[:, 4:].sum().sort_values()
    message_counts  = categories.values
    category_names = categories.index
    
    # create visuals
    graphs = [
        {
            'data': [
                Bar (
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
        },

        # plotting the number of messages for every category
        {
            'data': [
                Bar (
                    y=message_counts,
                    x=category_names,
                    )
            ],
           
            'layout': {
                'title': 'Number of Messages per Category',
                'yaxis': {
                    'title': "Messagecount",
                },
                'xaxis': {
                    'tickangle' : -45,
                },
            },
        },
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

    # This will render the go.html. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )

def main():
    app.run(host='0.0.0.0', port=3000, debug=True)

if __name__ == '__main__':
    main()