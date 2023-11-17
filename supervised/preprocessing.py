import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

def prepare_data():
    """
    Prepare data for ML training.

    Clean the dataset by
    1) Removing newline character, low information words
    2) Assign real review category of 0, computer generated review category of 1
    3) Change each word to lower case and its stem form

    Returns:
        data: panda dataframe
    """
    data = pd.read_csv('../fake reviews dataset.csv', names=['category', 'rating', 'label', 'text'])

    data['text'] = data['text'].replace('\n', '')
    # assign computer generated review with category of 1, real review with category of 0
    data['category'] = np.where(data['label'] == 'CG', 1, 0)

    data['text'] = punctuation_to_words(data)

    data.drop(index=data.index[0], axis=0, inplace=True)

    # download necessary files
    nltk.download('punkt')
    nltk.download('stopwords')

    data['text'] = data.apply(lambda x: tokenize(x['text']), axis=1)
    data['text'] = data.apply(lambda x: remove_stopWords(x['text']), axis=1)
    data['text'] = data.apply(lambda x: apply_stemming(x['text']), axis=1)

    data['text'] = data.apply(lambda x: rejoin_words(x['text']), axis=1)

    return data


def punctuation_to_words(data):
    """
    Convert punctuation to words.

    Punctuations like exclamation mark or question mark can be important in classification.

    Args:
        data (object): pandas dataframe 
    """
    
    data['text'] = data['text'].apply(lambda x: x.replace('!', ' exclamation '))
    data['text'] = data['text'].apply(lambda x: x.replace('?', ' question '))
    data['text'] = data['text'].apply(lambda x: x.replace('\'', ' quotation '))
    data['text'] = data['text'].apply(lambda x: x.replace('\"', ' quotation'))

    return data['text']

def tokenize(text):
    """
    Convert text to list of words.

    Args:
        text (string): input text

    Returns:
        list: list of words from text
    """

    tokens = nltk.word_tokenize(text)
    return [word for word in tokens if word.isalpha()]

def remove_stopWords(text):
    """
    Remove stopwords that gives low information.

    Ex: 'this', 'the', 'is'

    Args:
        text (list): input list

    Returns:
        list: list of words with stopwords removed
    """

    stop_words = set(stopwords.words("english"))
    return [word for word in text if word not in stop_words]

def apply_stemming(text):
    """
    Convert each word to its stemmed form using

    Ex: 'comfortable' -> 'comfort', 'information' -> 'inform'

    Args:
        text (list): input list

    Returns:
        list: list with stemmer applied
    """
    stemmer = PorterStemmer()
    return [stemmer.stem(word).lower() for word in text]

def rejoin_words(text):
    return (" ".join(text))

