import sys
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    '''
    load message and category dataset and merge them
    '''
    df_messages = pd.read_csv(messages_filepath)
    df_categories = pd.read_csv(categories_filepath)
    df = df_messages.merge(df_categories, how = 'inner', on = 'id')
    return df


def clean_data(df):
    '''
    process "categories" column by splitting it into seperate columns; 
    remove duplicated rows
    '''
    # split "categories" column into separate columns and save as a new dataframe
    category_names = [segment.split('-')[0] for segment in df["categories"][0].split(';')]
    df_splitted = df["categories"].str.split(';', expand = True).applymap(lambda x:int(x.split('-')[1]))
    df_splitted.columns = category_names

    # drop the original "categories" column 
    df = df.drop('categories', axis = 1)

    # concatenate the original dataframe with the new `categories` dataframe
    df_cat = pd.concat([df, df_splitted], axis = 1)

    # remove duplicated rows based on "id" and "message"
    df_nodup = df_cat.groupby(['id','message']).first().reset_index()
    return df_nodup


def save_data(df, database_filename):
    '''
    save the cleaned dataset into an sqlite database
    '''
    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('messages', engine, if_exists = 'replace', index = False)


def main():
    if len(sys.argv) == 4:
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]
        # load data
        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)
        # clean data
        print('Cleaning data...')
        df = clean_data(df)
        # save data to database
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)   
        print('Cleaned data saved to database!')
    else:
        print('''
            Please provide the filepaths of the messages and categories
            datasets as the first and second argument, respectively, \n 
            and the filename of the database to save the cleaned data
            to as the third argument. \n
            Example: python process_data.py disaster_messages.csv 
            disaster_categories.csv DisasterResponse.db
            ''')


if __name__ == '__main__':
    main()