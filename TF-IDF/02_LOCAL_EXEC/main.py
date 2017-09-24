"""
Author: Christos Hadjinikolis
Date:   21/05/2017
Desc:   Main Code
"""
# General IMPORTS ---------------------------------------------------------------------------------#
import codecs
import os
import pickle
import re
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt

# NLTK IMPORTS ------------------------------------------------------------------------------------#
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# SETTINGS IMPORTS --------------------------------------------------------------------------------#
from settings import DATA_DIR, ROOT, STOPWORDS_DIR

# PY-SPARK PATH SETUP AND IMPORTS -----------------------------------------------------------------#

# Path to source folder
SPARK_HOME = ROOT + '/spark-2.0.1-bin-hadoop2.7'
os.environ['SPARK_HOME'] = SPARK_HOME

# Append pyspark  to Python Path
sys.path.append(SPARK_HOME + '/python')
sys.path.append(SPARK_HOME + '/python/lib/py4j-0.10.3-src.zip')

try:
    from pyspark import SparkContext
    # from pyspark import SparkConf

    print 'Successfully imported Spark Modules'

except ImportError as error:
    print ('Can not import Spark Modules', error)
    sys.exit(1)

# GLOBAL VARIABLES --------------------------------------------------------------------------------#
SC = SparkContext('local[4]', 'TF-IDF Calculator')  # Instantiate a SparkContext object

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
DEBUG = False


# SUB-FUNCTIONS -----------------------------------------------------------------------------------#
def filter_word(raw_word):
    """

    :param raw_word:
    :return:
    """
    regex = re.compile(r'[^a-zA-Z]')

    uppercase_filtered_word = regex.sub("", raw_word)

    shortword = re.compile(r'\W*\b\w{1,3}\b')

    bigger_than_3_chars_word = shortword.sub('', uppercase_filtered_word)

    return bigger_than_3_chars_word


def remove_stopwords(not_checked_word):
    """

    :param not_checked_word:
    :return:
    """
    upper_word = not_checked_word.upper()

    for stopword in STOPWORDS:

        if upper_word == stopword:
            return None

    return upper_word


def apply_stemming(not_stemmed_word):
    """

    :param not_stemmed_word:
    :return:
    """
    return STEMMER.stem(not_stemmed_word)


def apply_lemmatization(not_lemmatized_word):
    """

    :param not_lemmatized_word:
    :return:
    """
    return WORDNET_LEMMATIZER.lemmatize(not_lemmatized_word)


def clean_word(raw_word):
    """

    :param raw_word:
    :return:
    """
    return remove_stopwords(apply_lemmatization(apply_stemming(filter_word(raw_word))))


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


# MAIN --------------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # LOAD ALL BOOKS AS A LIST OF LISTS -----------------------------------------------------------#

    # Get directories
    DIRECTORIES = [d for d in os.listdir(DATA_DIR)]
    DIRECTORIES.remove('README.md')

    if '.DS_Store' in DIRECTORIES:
        DIRECTORIES.remove('.DS_Store')

    print '\nBook dirs: ' + str(DIRECTORIES)

    if DEBUG:
        print 'To load books from: ' + ', '.join(DIRECTORIES)

    # Load books
    BOOKS_CONTENT = []   # List of books
    BOOKS_IDS = []       # to be used for a books name

    for directory in DIRECTORIES:

        books = [b for b in os.listdir(DATA_DIR + "/" + directory)]
        print "Loading " + str(len(books)) + " books for directory: " + directory + " ..."

        with tqdm(total=len(books)) as pbar:
            for book in books:

                # Add book ID
                BOOKS_IDS.append(book)

                if DEBUG:
                    print "\nParsing book: " + book

                try:
                    book_p = DATA_DIR + "/" + directory + "/" + book
                    with codecs.open(book_p, 'r', encoding='utf8') as book_lines:

                        book_content = []  # A book as a list of words

                        # I am doing the cleaning here to reduce the ammount of data that will be
                        # converted to an RDD.
                        for line in book_lines:
                            book_content.extend(line.split())

                        if DEBUG:
                            print "Parsing output: " + ", ".join(book_content)
                            raw_input("Press any key to continue to the next book:")

                        BOOKS_CONTENT.append([w for w in book_content if w is not None])

                except IOError:
                    print IOError.message
                    print IOError.filename
                pbar.update(1)
            print

    # Instantiate RDD
    print 'Create Parsed Books RDD'
    BOOKS_RDD = SC.parallelize(BOOKS_CONTENT)

    # Clean lines
    CLEANED_BOOKS_RDD = BOOKS_RDD.map(lambda ln: [clean_word(word) for word in ln])

    # Count Books to confirm they have all been successfully parsed
    NUMBER_OF_BOOKS = BOOKS_RDD.count()
    print 'Number of books added is ' + str(NUMBER_OF_BOOKS)

    # Create a dictionary -------------------------------------------------------------------------#
    print "----------------------------------------------------------------------------------------"
    raw_input('Produce IDF scores...')

    DICTIONARY_RDD_IDFS = idfs(BOOKS_RDD)
    UNIQUE_TOKEN_COUNT = DICTIONARY_RDD_IDFS.count()
    print 'There are %s unique tokens in the dataset.' % UNIQUE_TOKEN_COUNT

    IDF_TOKENS_SAMPLE = DICTIONARY_RDD_IDFS.takeOrdered(25, lambda s: s[1])
    print 'This is a dictionary sample of 25 words:'
    print '\n'.join(map(lambda (token, idf_score): '{0}: {1}'.format(token, idf_score),
                        IDF_TOKENS_SAMPLE))

    # Collect weights as a sorted map in descending order and save them
    DICTIONARY_RDD_IDFS_WEIGHTS = DICTIONARY_RDD_IDFS\
        .sortBy(lambda (token, score): score).collectAsMap()

    OUTPUT_1 = open('IDF-Weights/IDF-Weights.txt', "wb")
    OUTPUT_1.write("\n".join(map(lambda x: str(x), DICTIONARY_RDD_IDFS_WEIGHTS)))
    OUTPUT_1.close()

    # Write IDFS_weights_BV as python dictionary to a file
    OUTPUT_2 = open('IDF-Weights/dictionary_RDD_IDFs_Weights.pkl', 'wb')
    pickle.dump(DICTIONARY_RDD_IDFS_WEIGHTS, OUTPUT_2)
    OUTPUT_2.close()

    # CREATE A HISTOGRAM --------------------------------------------------------------------------#
    print '----------------------------------------------------------------------------------------'
    raw_input('Create an IDF-scores histogram...')
    IDFS_VALUES = DICTIONARY_RDD_IDFS\
        .map(lambda s: s[1]).collect()
    FIG = plt.figure(figsize=(8, 3))
    plt.hist(IDFS_VALUES, 50, log=True)
    plt.show()

    # SAVE PARSED BOOKS ---------------------------------------------------------------------------#
    print '----------------------------------------------------------------------------------------'
    raw_input('Save parsed books ...')
    # Add keys to parsed books
    BOOKS_KV_RDD = SC.parallelize([BOOKS_IDS, BOOKS_RDD.collect()]).collect()

    # Write books_KV_RDD as python dictionary to a file
    OUTPUT_3 = open('parsed-books/books_KV_RDD.pkl', 'wb')
    pickle.dump(BOOKS_KV_RDD, OUTPUT_3)
    OUTPUT_3.close()

# END OF FILE -------------------------------------------------------------------------------------#
