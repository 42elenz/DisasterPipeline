import sys
import pandas as pd
from sqlalchemy.engine import create_engine

def load_data(messages_filepath, categories_filepath):
    # load messages and categories dataset
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    # merge datasets
    df = pd.merge(categories, messages, on=['id'], how='outer')
    return df

def drop_zero_variance(df):
    column_names = df.columns
    for column in column_names:
        if len(df[column].unique()) < 2:
               df = df.drop(columns=column)
    return df

def clean_data(df):
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0, :]
    # use this row to extract a list of new column names for categories.
    category_colnames = row.apply(lambda x: x[:-2]).values

    # rename the columns of `categories`
    categories.columns = category_colnames
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x[-1])
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    # drop the original categories column from `df`
    df = df.drop(columns='categories')

    # dropping features with 0 variance
    categories = drop_zero_variance(categories)
    
    # replacing 'related'-feature values of 2 with 0
    categories['related'] = categories['related'].replace(2, 0)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df_new = pd.concat([df, categories], axis=1)

    df_cleaned = df_new[~(df_new["id"].duplicated())]
  
    return df_cleaned

def save_data(df, database_filename):
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()