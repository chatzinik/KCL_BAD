"""
Author: Christos Hadjinikolis, Satyasheel
Date:   05/02/2017
Desc:   Data pre-processing.
"""
# LIBRARIES ---------------------------------------------------------------------------------------#
import os
import re
import sys

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer

from settings import DATA
from settings import ROOT

# PYSPARK PATH SETUP AND IMPORTS ------------------------------------------------------------------#
os.environ['SPARK_HOME'] = os.path.join(ROOT, 'spark/spark-2.1.0-bin-hadoop2.7')

# APPEND PySPARK TO PYTHON PATH
sys.path.append(os.path.join(os.environ['SPARK_HOME'], "python"))
sys.path.append(os.path.join(os.environ['SPARK_HOME'], "python/lib/py4j-0.10.4-src.zip"))

try:
    from pyspark import SparkContext
    from pyspark import SparkConf
    from pyspark.sql import Row
    from pyspark.sql import SQLContext
    from pyspark.accumulators import AccumulatorParam

    print("Successfully imported Spark Modules")

except ImportError as e:
    print("Can not import Spark Modules", e)
    sys.exit(1)

# GLOBAL VARIABLES --------------------------------------------------------------------------------#
sc = SparkContext('local[4]', 'data_prep')  # Instantiate a SparkContext object
sqlContext = SQLContext(sc)  # Instantiate a sqlContext object


# FUNCTIONS ---------------------------------------------------------------------------------------#
def filter_sentence(sentence):
    # Remove handles/hastags
    sentence = re.sub("([@|#].*?)", " ", sentence)

    # Remove floating point numbers
    sentence = re.sub("([/| |'|(|+|-]\d+[\.| |/|;|?|%|:|,|'|(|)|+|-]\d*.?)", " ", sentence)

    # Remove numbers!
    sentence = re.sub("( \d+.? )", " ", sentence)

    # Remove additional abnormalities
    sentence = re.sub("([ |.]\d+[-|\.]\d*.? )", " ", sentence)
    sentence = re.sub("(\d+-\d+.?)", "", sentence)

    return sentence


def lemmatize(sentence_words):
    # Instantiate lemmatisation-object
    wordnet_lemmatizer = WordNetLemmatizer()

    # Lemmatizer:
    # lowering stinr is necessary in this step - unfortunately it strips words of semantics
    # (e.g CAPITALS are usually used for shouting!)
    for i in range(len(sentence_words)):
        sentence_words[i] = wordnet_lemmatizer.lemmatize(sentence_words[i].lower())

    return sentence_words


# MAIN CODE ---------------------------------------------------------------------------------------#
if __name__ == "__main__":

    # Data Prep Step 1 ----------------------------------------------------------------------------#
    RAW_PATH = "/raw/cornell_movie_dialogs_corpus/movie_lines.txt"
    with open(DATA + RAW_PATH) as f:
        content = f.readlines()

    with open(DATA + "/pre-processed/utterances.txt", "a") as fp:
        for line in content:
            new_line_1 = line.replace('\t', " ")
            new_line_2 = new_line_1.split(" +++$+++ ")
            new_line_3 = '\t'.join(new_line_2)
            fp.write(new_line_3)

    # Load and process corpus ---------------------------------------------------------------------#
    INPUT = DATA + "/pre-processed/utterances.txt"
    tokenizer = RegexpTokenizer(r'(\w+)')

    data_RDD = sc.textFile(INPUT)
    column_data_RDD = data_RDD.map(lambda row: row.split("\t"))

    # Collect sentences
    corpus_data_RDD = (column_data_RDD
                       .map(lambda row: row[4])  # Text is in the fourth list element.
                       .map(filter_sentence)  # Remove redundant or distorted info.
                       .map(tokenizer.tokenize)  # Convert every line into a list of words.
                       .map(lemmatize))  # Apply lemmatization.

    corpus_data = corpus_data_RDD.collect()

    with open(DATA + "/pre-processed/utterances2listsOfwords.txt", "a") as fp:
        for line in corpus_data:
            fp.write(",".join(line) + "\n")

# END OF FILE -------------------------------------------------------------------------------------#
