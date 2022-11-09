# Disaster Response Pipeline Project

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Go to `app` directory: `cd app`


3. Run your web app: `python run.py`



4. Click the `PREVIEW` button to open the homepage


Discussion:
Looking at the table ... it can be deducted that there is a inbalance in the distrubtions of our classes. This leads to two main problems:
1. Evalutation of the model outcomes
2. Training of our model

Refering to the first point it can be said that the accuracy, which is often used to evaluate a model, can not be used without constraints. It may be good enough for a well-balanced class but not ideal for the imbalanced class problem. 

For an imbalanced class dataset F1 score is a more appropriate metric. It is the harmonic mean of precision and recall.
If the classifier predicts the minority class but the prediction is erroneous and false-positive increases, the precision metric will be low and so as F1 score. Also, if the classifier identifies the minority class poorly, i.e. more of this class wrongfully predicted as the majority class then false negatives will increase, so recall and F1 score will low. F1 score only increases if both the number and quality of prediction improves.
Therefor I used the F1 score to evaluate the models performance. 

With reference to the second point it must be  taken into account that the minority class is harder to predict because there are few examples of this class, by definition. This means it is more challenging for a model to learn the characteristics of examples from this class, and to differentiate examples from this class from the majority classes.