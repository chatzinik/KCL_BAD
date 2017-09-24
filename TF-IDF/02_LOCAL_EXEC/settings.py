"""
Author: Christos Hadjinikolis, Satyasheel
Date:   21/01/2017
Desc:   Path configuration settings.
"""
# IMPORTS -----------------------------------------------------------------------------------------#
import os

# PATHS to ROOT and DATA Dirs
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, '../External_DS/books')
STOPWORDS_DIR = os.path.join(ROOT, '../External_DS/stopwords')

# END OF FILE -------------------------------------------------------------------------------------#
