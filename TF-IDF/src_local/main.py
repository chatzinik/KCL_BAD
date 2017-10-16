"""
Author: Christos Hadjinikolis
Date:   21/05/2017
Desc:   Main Code
"""
# General IMPORTS -----------------------------------------------------------------------------------------------------#
import codecs
import os
import pickle
import re
import sys
import json
from tqdm import tqdm
import matplotlib.pyplot as plt

# NLTK IMPORTS --------------------------------------------------------------------------------------------------------#
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# SETTINGS IMPORTS ----------------------------------------------------------------------------------------------------#
from settings.paths import DATA_DIR, ROOT, STOPWORDS_DIR

# PY-SPARK PATH SETUP AND IMPORTS -------------------------------------------------------------------------------------#

# Path to source folder
SPARK_HOME = ROOT + '/spark-2.2.0-bin-hadoop2.7'
os.environ['SPARK_HOME'] = SPARK_HOME

# Append pyspark  to Python Path
sys.path.append(SPARK_HOME + '/python')
sys.path.append(SPARK_HOME + '/python/lib/py4j-0.10.4-src.zip')

try:
    from pyspark import SparkContext
    from pyspark import SparkConf

    print('Successfully imported Spark Modules')

except ImportError as error:
    print('Can not import Spark Modules', error)
    sys.exit(1)

# GLOBAL VARIABLES ----------------------------------------------------------------------------------------------------#
conf = SparkConf().setAppName("TF-IDF Calculator")
conf = (conf.setAppName("App")
        .setMaster('local[*]')
        .set('spark.executor.memory', '8G')
        .set('spark.driver.memory', '24G')
        .set('spark.driver.maxResultSize', '12'))

SC = SparkContext(conf=conf) # Instantiate a SparkContext object

# Load stopwrds
STOPWORDS = []
with codecs.open(STOPWORDS_DIR + '/stopwords.txt', 'r', encoding='UTF-8') as f:
    for line in f:
        cleanedLine = line.strip()
        if cleanedLine:  # is not empty
            STOPWORDS.append(cleanedLine)

# Specify nltk data path
nltk.data.path.append('/Users/christoshadjinikolis/Applications/nltk_data/')

# initialise Stemmer and Lemmatizer
STEMMER = SnowballStemmer('english')
WORDNET_LEMMATIZER = WordNetLemmatizer()

# Debugging
DEBUG = True


# SUB-FUNCTIONS -------------------------------------------------------------------------------------------------------#
def filter_word(word):
    """
    A function that filters words.
    :param word: A word token (String)
    :return: A filtered word token (String)
    """

    # Remove non-alphabetical cars
    regex = re.compile(r'[^a-zA-Z]')
    filtered_word = regex.sub("", word)

    # Return only words that are bigger than 3 characters
    if len(filtered_word) > 3:
        return filtered_word
    else:
        return None


def remove_stopwords(word):
    """
    A function that removes stopwords
    :param word: A word token (String)
    :return: The same word if it is not a stopword (String).
    """

    for stopword in STOPWORDS:
        if word == stopword:
            return None
    return word


def apply_stemming(word):
    """

    :param word:
    :return:
    """
    return STEMMER.stem(word)


def apply_lemmatization(word):
    """

    :param word:
    :return:
    """
    return WORDNET_LEMMATIZER.lemmatize(word)


def clean_word(word):
    """

    :param word:
    :return:
    """
    filtered_word = filter_word(word)

    if filtered_word is None:
        return None
    else:
        return remove_stopwords(apply_lemmatization(apply_stemming(filtered_word)).upper())


def tf(tokens):
    """ Compute TF
    Args:
        tokens (list of str): input list of tokens from tokenize
    Returns:
        dictionary: a dictionary of tokens to its TF values
    """
    counts = {}
    number_of_words = len(tokens)

    for t in tokens:
        counts.setdefault(t, 0.0)
        counts[t] += 1
    return {t: counts[t] / number_of_words for t in counts}


def idfs(corpus):
    """
    A function for computing IDF socres
    :param corpus: an RDD of parsed books into list of words, capitalised and cleaned.
    :return: an RDD of (token, IDF value)
    """

    number_of_instances = corpus.count()

    # The result of the next line will be a list with distinct tokens...

    # No more records! FLATMAP --> unique_tokens is ONE SINGLE LIST
    unique_tokens = corpus.flatMap(lambda x: list(set(x)))

    # every element in the list will become a pair!
    token_count_pair_tuple = unique_tokens.map(lambda x: (x, 1))

    # same elements in lists are aggregated
    token_sum_pair_tuple = token_count_pair_tuple.reduceByKey(lambda a, b: a + b)

    # compute weight
    return token_sum_pair_tuple.map(lambda x: (x[0], float(number_of_instances) / x[1]))


# MAIN ----------------------------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # LOAD ALL BOOKS AS A LIST OF LISTS -------------------------------------------------------------------------------#

    # Get directories
    DIRECTORIES = [d for d in os.listdir(DATA_DIR)]
    DIRECTORIES.remove('README.md')

    if '.DS_Store' in DIRECTORIES:
        DIRECTORIES.remove('.DS_Store')

    print('\nBook dirs: ' + str(DIRECTORIES))

    if DEBUG:
        print('To load books from: ' + ', '.join(DIRECTORIES))

    # Load books
    BOOKS_CONTENT = []   # List of books
    BOOKS_IDS = []       # to be used for a books name

    for directory in DIRECTORIES:

        books = [b for b in os.listdir(DATA_DIR + "/" + directory)]
        print("Loading " + str(len(books)) + " books for directory: " + directory + " ...")

        with tqdm(total=len(books)) as pbar:
            for book in books:

                # Add book ID
                BOOKS_IDS.append(book)

                try:
                    book_p = DATA_DIR + "/" + directory + "/" + book
                    with codecs.open(book_p, 'r', encoding='utf8') as book_lines:

                        book_content = []  # A book as a list of words

                        # I am doing the cleaning here to reduce the amount of data that will be
                        # converted to an RDD.
                        for line in book_lines:
                            book_content.extend(line.split())

                        BOOKS_CONTENT.append([w for w in book_content if w is not None])

                except IOError:
                    print(IOError.message)
                    print(IOError.filename)
                pbar.update(1)
            print()

    # Instantiate RDD -------------------------------------------------------------------------------------------------#
    print('Create Parsed Books RDD')
    BOOKS_RDD = SC.parallelize(BOOKS_CONTENT, numSlices=2500)

    # Count Books to confirm they have all been successfully parsed
    NUMBER_OF_BOOKS = BOOKS_RDD.count()
    print('Number of books added is ' + str(NUMBER_OF_BOOKS) + ", Partitions: " + str(BOOKS_RDD.getNumPartitions()))

    # Clean lines
    CLEANED_BOOKS_RDD = BOOKS_RDD\
        .map(lambda ln: list(filter(None.__ne__, [clean_word(word) for word in ln])))

    # Create a dictionary ---------------------------------------------------------------------------------------------#
    print("----------------------------------------------------------------------------------------")
    print('Produce IDF scores...')

    DICTIONARY_RDD_IDFS = idfs(CLEANED_BOOKS_RDD)
    print('There are %s unique tokens in the dataset.' % DICTIONARY_RDD_IDFS.count())

    IDF_TOKENS_SAMPLE = DICTIONARY_RDD_IDFS.takeOrdered(50, lambda s: -s[1])
    print('This is a dictionary sample of 25 words:')
    print('\n'.join(map(lambda tuple_xx: '{0}: {1}'.format(tuple_xx[0], tuple_xx[1]), IDF_TOKENS_SAMPLE)))

    # Collect weights as a sorted map in descending order and save them
    DICTIONARY_RDD_IDFS_WEIGHTS = (DICTIONARY_RDD_IDFS
                                   .sortBy(lambda tuple_xx: tuple_xx[1], ascending=False)
                                   .collectAsMap())

    json.dump(DICTIONARY_RDD_IDFS_WEIGHTS, open("idf_weights/idf_words.json", 'w'), indent=2)

    # Write IDFS_weights_BV as python dictionary to a file
    OUTPUT_1 = open('idf_weights/dictionary_RDD_IDFs_Weights.pkl', 'wb')
    pickle.dump(DICTIONARY_RDD_IDFS_WEIGHTS, OUTPUT_1)
    OUTPUT_1.close()

    # SAVE PARSED BOOKS -----------------------------------------------------------------------------------------------#
    print('----------------------------------------------------------------------------------------')
    print('Save parsed books ...')
    # Add keys to parsed books
    BOOKS_KV_RDD = SC.parallelize([BOOKS_IDS, CLEANED_BOOKS_RDD.collect()], numSlices=2500)\
        .collectAsMap()

    # Write books_KV_RDD as python dictionary to a file (use wb as this is a serialised binary file)
    OUTPUT_2 = open('parsed_books/books_KV_RDD.pkl', 'wb')
    pickle.dump(BOOKS_KV_RDD, OUTPUT_2)
    OUTPUT_2.close()

    # CREATE A HISTOGRAM ----------------------------------------------------------------------------------------------#
    print('----------------------------------------------------------------------------------------')
    print('Create an IDF-scores histogram...')

    IDFS_VALUES = (DICTIONARY_RDD_IDFS_WEIGHTS
        .map(lambda s: s[1])
        .sortBy(lambda s: s, ascending=False)
        .collect())

    FIG = plt.figure(figsize=(8, 3))
    plt.hist(IDFS_VALUES, 50, log=True)
    plt.show()

# END OF FILE ---------------------------------------------------------------------------------------------------------#
