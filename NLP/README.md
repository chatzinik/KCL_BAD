# KCL_Research_NLP
Repo for data and code for our research. 

## How to run the code
Use the Python 2.7 Interpreter. 

Also, I used Apache Spark for cleaning the data. You want be able to run the code without spark on your 
machine. If you want to run the code on your machine download spark from [here](http://spark.apache.org/downloads.html) 
and include in the project directory as shown in the picture below.

![Directory Image](imgs/spark_file.png?raw=true "Directory structure.")

## Notice
All words were lowerised in order to apply the WordNet lemmatizer. Maybe we should repeat the 
preprocessing step without lemmatisation. Also, stop words have not been removed. Additional preprocessing 
may be necessary to remove single character words like "s" or "I". Hyphen's, quotations etc have also been 
removed (check the `filter_sentences` function).

## Useful Links

1. [Dive Into NLTK, Part IV: Stemming and Lemmatization](http://textminingonline.com/dive-into-nltk-part-iv-stemming-and-lemmatization)
2. [Stop Words](https://en.wikipedia.org/wiki/Stop_words)
3. [Python NLTK Toolkit](http://www.nltk.org/)