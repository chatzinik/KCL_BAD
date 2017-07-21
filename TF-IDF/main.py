"""
Author: Christos Hadjinikolis
Date:   21/05/2017
Desc:   Main Code
"""
# General IMPORTS ---------------------------------------------------------------------------------#
import os
import pickle
import re
import sys
import codecs
import nltk
from tqdm import *
import matplotlib.pyplot as plt

# NLTK IMPORTS ------------------------------------------------------------------------------------#
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer

# SETTINGS IMPORTS --------------------------------------------------------------------------------#
from settings import ROOT
from settings import DATA_DIR
from settings import STOPWORDS_DIR

# PY-SPARK PATH SETUP AND IMPORTS -----------------------------------------------------------------#

# Path to source folder
SPARK_HOME = ROOT + "/spark-2.0.1-bin-hadoop2.7"
os.environ['SPARK_HOME'] = SPARK_HOME

# Append pyspark  to Python Path
sys.path.append(SPARK_HOME + "/python")
sys.path.append(SPARK_HOME + "/python/lib/py4j-0.10.3-src.zip")

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import Row
    from pyspark.sql import SQLContext
    from pyspark.mllib.linalg import SparseVector
    from pyspark.accumulators import AccumulatorParam

    print ("Successfully imported Spark Modules")

except ImportError as e:
    print ("Can not import Spark Modules", e)
    sys.exit(1)

# GLOBAL VARIABLES --------------------------------------------------------------------------------#
sc = SparkContext('local[4]', 'TF-IDF Calculator')  # Instantiate a SparkContext object
sqlContext = SQLContext(sc)  # Instantiate a sqlContext object

# Load stopwrds
STOPWORDS = []
with codecs.open(STOPWORDS_DIR + "/stopwords.txt", 'r', encoding='UTF-8') as f:
    for line in f:
        cleanedLine = line.strip()
        if cleanedLine:  # is not empty
            STOPWORDS.append(cleanedLine)

# Specify nltk data path
nltk.data.path.append('/Users/christoshadjinikolis/Applications/nltk_data/')

# initialise Stemmer and Lemmatizer
STEMMER = SnowballStemmer("english")
WORDNET_LEMMATIZER = WordNetLemmatizer()


# SUB-FUNCTIONS -----------------------------------------------------------------------------------#
def filter_word(raw_word):
    uppercase_filtered_word = re.sub("([!?\"./*-_();:[]{}|~])", "", raw_word)

    return uppercase_filtered_word


def remove_stopwords(not_checked_word):
    upper_word = not_checked_word.upper()

    for stopword in STOPWORDS:

        if upper_word == stopword:
            return None

    return upper_word


def apply_stemming(not_stemmed_word):
    return STEMMER.stem(not_stemmed_word)


def apply_lemmatization(not_lemmatized_word):
    return WORDNET_LEMMATIZER.lemmatize(not_lemmatized_word)


def clean_word(raw_word):
    return remove_stopwords(apply_lemmatization(apply_stemming(filter_word(raw_word))))


def idfs(corpus):
    """ Compute IDF
    Args:
        corpus (RDD): input corpus
    Returns:
        RDD: a RDD of (token, IDF value)
    """

    N = corpus.count()

    # The result of the next line will be a list with distinct tokens...

    # No more records! FLATMAP --> unique_tokens is ONE SINGLE LIST
    unique_tokens = corpus.flatMap(lambda x: list(set(x)))

    # every element in the list will become a pair!
    token_count_pair_tuple = unique_tokens.map(lambda x: (x, 1))

    # same elements in lists are aggregated
    token_sum_pair_tuple = token_count_pair_tuple.reduceByKey(lambda a, b: a + b)

    # compute weight
    return token_sum_pair_tuple.map(lambda x: (x[0], float(N) / x[1]))

# MAIN --------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # LOAD ALL BOOKS AS A LIST OF LISTS -----------------------------------------------------------#

    # Get directories
    directories = [d for d in os.listdir(DATA_DIR)]

    if '.DS_Store' in directories:
        directories.remove('.DS_Store')

    # Load books
    books_content = []  # List of books
    for directory in directories:

        books = [b for b in os.listdir(DATA_DIR + "/" + directory)]
        print("Loading " + str(len(books)) + " books for directory: " + directory + " ...")

        with tqdm(total=len(books)) as pbar:
            for book in books:
                try:
                    book_p = DATA_DIR + "/" + directory + "/" + book
                    with codecs.open(book_p, 'r', encoding='utf8') as book_lines:

                        book_content = []  # A book as a list of words
                        for line in book_lines:
                            book_content.extend([clean_word(word) for word in line.split()])
                        books_content.append([w for w in book_content if w is not None])

                except IOError:
                    print IOError.message
                    print IOError.filename
                pbar.update(1)
        print

    # Instantiate RDD
    books_RDD = sc.parallelize(books_content)

    # Count Books to confirm they have all been successfully parsed
    number_of_books = books_RDD.count()

    # Create a dictionary -------------------------------------------------------------------------#
    print("---------------------------------------------------------------------------------------")
    raw_input("Produce TF-IDF scores...")

    dictionary_RDD_IDFs = idfs(books_RDD)
    unique_token_count = dictionary_RDD_IDFs.count()
    print 'There are %s unique tokens in the dataset.' % unique_token_count

    IDFS_Tokens_Sample = dictionary_RDD_IDFs.takeOrdered(25, lambda s: s[1])
    print("This is a dictionary sample of 25 words:")
    print '\n'.join(map(lambda (token, idf_score): '{0}: {1}'.format(token, idf_score),
                        IDFS_Tokens_Sample))

    # Create a broadcast variable for the weighted dictionary (sorted)
    dictionary_RDD_IDFs_Weights = dictionary_RDD_IDFs\
        .sortBy(lambda (token, score): score).collectAsMap()
    IDFS_weights_BV = sc.broadcast(dictionary_RDD_IDFs_Weights)

    # Write IDFS_weights_BV as python dictionary to a file
    output = open('/Users/path/to/dictionary_RDD_IDFs_Weights.pkl', 'wb')
    pickle.dump(dictionary_RDD_IDFs_Weights, output)
    output.close()

    print IDFS_weights_BV.value

    # CREATE A HISTOGRAM --------------------------------------------------------------------------#
    print("---------------------------------------------------------------------------------------")
    raw_input("Create an IDF-scores histogram...")
    IDFs_values = dictionary_RDD_IDFs.map(lambda s: s[1]).collect()
    fig = plt.figure(figsize=(8, 3))
    plt.hist(IDFs_values, 50, log=True)
    plt.show()

# END OF FILE -------------------------------------------------------------------------------------#

