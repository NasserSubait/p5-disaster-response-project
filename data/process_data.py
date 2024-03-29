import sys
import pandas as pd
import numpy as np
import sqlite3

def load_data(messages_filepath, categories_filepath):
    
    '''
    loading data from two different files and merg them 

    paramerter
    -------------
    messages_filepath: file path [str]
    categories_filepath: file path [str]

    return
    ------------
    df: a dataframe

    ''' 
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = pd.merge(messages,categories,on='id')
    
    return df

    
def clean_data(df):
    
    '''
    clean the data

    paramerter
    -------------
    df: dataframe

    return
    ------------
    df: cleaned data frame 

    ''' 

    categories = df['categories'].str.split(';',expand=True)
    row = categories.iloc[0].apply(lambda x: x.split('-')[0])
    category_colnames = row.to_list()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[1])
    
        # convert column from string to numeric
        categories[column] = categories[column].apply(lambda x: int(x))
    
    df.drop('categories',axis=1,inplace=True)
    df = pd.concat([df,categories],axis=1)

    return df


def save_data(df, database_filename):

    '''
    save the data to a sqlite database

    paramerter
    -------------
    df: dataframe that need to be saved
    database_filename: the name of the database [str]

    return
    ------------
    None 

    ''' 

    conn = sqlite3.connect(database_filename)  
    df.to_sql('messages',con = conn, if_exists='replace',index=False)


def main():

    '''
    the start of the program the system input will take the file name of
    messages, categories and database and execute other functions on them 

    paramerter
    -------------
    None

    return
    ------------
    None 

    ''' 
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