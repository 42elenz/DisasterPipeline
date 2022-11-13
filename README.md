### Project Description
    The goal of this project was to build a model to classfiy twitter messages that are send during disasters. Messages will be categorized into 35 pre-defined categories, such as Aid Related, Medical Help, Search And Rescue. As a result the messages can be routed to the appropriate disaster relief agencies.

    The data set was provided by Figure Eight containing real messages that were sent during disaster events. Steps included building a basic ETL and Machine Learning pipeline as well as a webapp. Text preprocessing was done using tokenizing, stemming and lemmatizing. The multi-label classifier was built using the pipeline features of Scikit Learn. Grid Search Cross Validation was used to tune the hyperparameters.
    The web app is able to process textmessages and classify them according according to the model and display statistics using graphical plots.

### File Descriptions
    data/process_data.py - The ETL script
    models/train_classifier - The ML script
    app/run.py - The server for the website
    app/templates - The website HTML/CSS files
    data/*.csv - The dataset

### Installation
    Run pip install -r requirements.txt

### Instructions:
    1. Run the following commands in the project's root directory to set up your database and model.
        To run ETL pipeline that cleans data and stores in database python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
        To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl
    2. Run the following command in the app's directory to run your web app. python run.py
    3. Go to http://0.0.0.0:3001/

### Further Discussion:
    Initial exploration of the data showed inbalance in the distrubtions of the classes. 
    
    This leads to two main problems:
    1. Evalutation of the model outcomes
    2. Training of our model

    1. Accuracy, can not be used to evaluate model predictions on imbalanced classes since it does not distinguish between the numbers of correctly classified examples of different classes. For an imbalanced class dataset F1 score is a more appropriate metric. It is the harmonic mean of precision and recall. If the classifier predicts the minority class but the prediction is erroneous and false-positive increases, the precision metric will be low and so is F1 score. Also, if the classifier identifies the minority class poorly, i.e. more of this class wrongfully predicted as the majority class then false negatives will increase, so recall and F1 score will low. F1 score only increases if both the number and quality of prediction improves. Therefor I used the F1 score to evaluate the models performance. 

    2. The minority class is harder to predict because there are few examples of this class, by definition. This means it is more challenging for a model to learn the characteristics of examples from this class, and to differentiate examples from this class from the majority classes.
    
### Acknowledgements
    Udacity Data Scientist Nanodegree Program