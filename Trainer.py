#Trains the nltk taggers.

from pickle import dump
import nltk
from nltk.tokenize import word_tokenize
import re

training_set = [[(word.lower(), tagged) for word, tagged in sentence] for sentence in
                nltk.corpus.conll2002.tagged_sents('esp.train')]
##tagged corpus from conll2002
training_set = [sentence for sentence in nltk.corpus.conll2002.tagged_sents('esp.train')]

##reggex patterns to morphologically correct misses from bigram tagger.
patterns = [(r'.*ar$','VI'),
            #(r'.*er$','VI'),
            (r'.*ir$','VI'),
            (r'.*ando$','VG'),
            (r'.*endo$','VG'),
            (r'.*ó$','VPP'),
            (r'^http.*','IGNORE'),
            (r'^#.*','IGNORE'),
            (r'^//.*','IGNORE')]
##all elements not founded are most likely Nouns.
default_tagger = nltk.DefaultTagger('NN')
unigram_tagger = nltk.UnigramTagger(training_set, backoff=default_tagger)
bigram_tagger = nltk.BigramTagger(training_set, backoff=unigram_tagger)
regex_tagger = nltk.RegexpTagger(patterns, backoff=bigram_tagger)

output = open('training/regex_tagger.pkl','wb')
dump(regex_tagger,output,-1)
output.close()