"""
Author: Christos Hadjinikolis, Satyasheel
Date:   21/01/2017
Desc:   Path configuration settings.
"""
# IMPORTS -----------------------------------------------------------------------------------------#
import os

# PATHS to ROOT and DATA Dirs
ROOT = os.path.abspath(os.path.dirname(__file__))
DATA_DIR = os.path.join(ROOT, 'books')
STOPWORDS_DIR = os.path.join(ROOT, 'stopwords')

# END OF FILE -------------------------------------------------------------------------------------#
