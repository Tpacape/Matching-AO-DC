#!/usr/bin/env python
"""
coding=utf-8

Code Template

"""
import inspect
import logging
import os
import sys

import pandas
import spacy

from matplotlib import pyplot as plt

# Collections
import collections
from collections import Counter

# NLTK
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models.tfidfmodel import TfidfModel

# Spacy
import spacy

# Polyglot
from polyglot.text import Text


currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from bin import field_extraction
from bin import lib

# Pycharm 3.6
# Anaconda 3.6
# Save en 2 CSV row pour charger plus facilement
# Notebook plutot
# pytorch / spacy / nltk
# prepro diffé&rent pour chaque doc
# Counter + TSNE + k-means (elbow) + 

def main():
    """
    Main function documentation template
    :return: None
    :rtype: None
    """
    logging.getLogger().setLevel(logging.INFO)

    # Extract data from upstream.
    observations = extract()

    # Transform data to have appropriate fields
    #observations = transform(observations)
    
    observations = preprocessingNLTK(observations)
    
    
    observations = preprocessingGensim(observations)
    
    
    observations = preprocessingSpacy(observations)
    
    # Save data for downstream consumption
    #save(observations)

    pass

def extract():
    logging.info('Begin extract')

    # Reference variables
    candidate_file_agg = list()

    # Create list of candidate files
    for root, subdirs, files in os.walk(lib.get_conf('resume_directory')):
        folder_files = map(lambda x: os.path.join(root, x), files)
        candidate_file_agg.extend(folder_files)

    # Convert list to a pandas DataFrame
    observations = pandas.DataFrame(data=candidate_file_agg, columns=['file_path'])
    logging.info('Found {} candidate files'.format(len(observations.index)))

    # Subset candidate files to supported extensions
    observations['extension'] = observations['file_path'].apply(lambda x: os.path.splitext(x)[1])
    observations = observations[observations['extension'].isin(lib.AVAILABLE_EXTENSIONS)]
    logging.info('Subset candidate files to extensions w/ available parsers. {} files remain'.
                 format(len(observations.index)))

    # Attempt to extract text from files
    observations['text'] = observations['file_path'].apply(lib.convert_pdf)
    
    logging.info('End extract')
    return observations


def transform(observations):
    logging.info('Begin transform')

    # Extract skills
    observations = field_extraction.extract_fields(observations)

    logging.info('End transform')
    return observations

def preprocessingNLTK(observations):
    logging.info('Begin preprocessingNLTK')
    
    
    observations['tokens'] = ""
    observations['no_stops'] = ""
    observations['lemmatized'] = ""
    
    # Instantiate the WordNetLemmatizer
    wordnet_lemmatizer = WordNetLemmatizer()
        
    for index, row in observations.iterrows():
        
        # Tokenization
        tokens = [w for w in word_tokenize(row["text"].lower())
        if w.isalpha()]
        observations.loc[index, 'tokens'] = tokens
        
        # del stop words
        no_stops = [w for w in tokens
                     if w not in stopwords.words('french')]
        observations.loc[index, 'no_stops'] = no_stops
        
        # Lemmatize all tokens into a new list: lemmatized
        lemmatized = [wordnet_lemmatizer.lemmatize(w) for w in no_stops]
        observations.loc[index, 'lemmatized'] = lemmatized
        
    
    logging.info('End preprocessingNLTK')
    return observations


def preprocessingGensim(observations):
    logging.info('Begin preprocessingGensim')
    
    
    observations['tf-idf'] = ""

    
    # Create a Corpus
    dictionary = Dictionary(observations["lemmatized"].tolist())
    corpus = [dictionary.doc2bow(text) for text in observations['lemmatized'].tolist()]
        
    # Create a new TfidfModel using the corpus
    tfidf = TfidfModel(corpus)
    
    for index, row in observations.iterrows():
        
        tfidf_weights = tfidf[corpus[index]]
        sorted_tfidf_weights = sorted(tfidf_weights, key=lambda w: w[1], reverse=True)
        observations.loc[index, 'tf-idf'] = sorted_tfidf_weights
        

    logging.info('End preprocessingGensim')
    return observations

def preprocessingSpacy(observations):
    logging.info('Begin preprocessingSpaCy')
    
    
    observations['nlp'] = ""
    
    # Instantiate the nlp
    nlp = spacy.load('en_core_web_sm')
        
    for index, row in observations.iterrows():
        
        # Tokenization
        nlp = nlp(observations.loc[index, 'text'])
        observations.loc[index, 'nlp'] = nlp

    
    logging.info('End preprocessingSpaCy')
    return observations


def save(observations):
    logging.info('Begin load')
    
    output_path = os.path.join(lib.get_conf('summary_output_directory'), 'resume_summary.csv')
    
    tmp = observations.copy()
    del tmp['text']
    tmp.to_csv(path_or_buf=output_path, index_label='index')
    
    logging.info('End transform')
    pass


# Main section
if __name__ == '__main__':
    main()
