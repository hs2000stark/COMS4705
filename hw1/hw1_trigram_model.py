#!/usr/bin/env python
# coding: utf-8

# In[19]:


import sys
from collections import defaultdict
import math
import random
import os
import os.path
"""
COMS W4705 - Natural Language Processing - Fall 2023 
Programming Homework 1 - Trigram Language Models
Daniel Bauer
"""

def corpus_reader(corpusfile, lexicon=None): 
    with open(corpusfile,'r') as corpus: 
        for line in corpus: 
            if line.strip():
                sequence = line.lower().strip().split()
                if lexicon: 
                    yield [word if word in lexicon else "UNK" for word in sequence]
                else: 
                    yield sequence

def get_lexicon(corpus):
    word_counts = defaultdict(int)
    for sentence in corpus:
        for word in sentence: 
            word_counts[word] += 1
    return set(word for word in word_counts if word_counts[word] > 1)  



def get_ngrams(sequence, n):
    # Add START and STOP padding based on the n-gram size
    start_padding = ['START'] * (n-1) if n > 1 else ['START']
    stop_padding = ['STOP']
    padded_tokens = start_padding + sequence + stop_padding
    
    # Generate n-grams
    ngrams = [tuple(padded_tokens[i:i+n]) for i in range(len(padded_tokens) - n + 1)]
    
    return ngrams



class TrigramModel(object):
    num_sentence = 41614 # total counts of sentence (equal to the count of ('START') in unigramcounts)
    
    
    def __init__(self, corpusfile):
    
        # Iterate through the corpus once to build a lexicon 
        generator = corpus_reader(corpusfile)
        self.lexicon = get_lexicon(generator)
        self.lexicon.add("UNK")
        self.lexicon.add("START")
        self.lexicon.add("STOP")
    
        # Now iterate through the corpus again and count ngrams
        generator = corpus_reader(corpusfile, self.lexicon)
        self.count_ngrams(generator)


    def count_ngrams(self, corpus):

   
        self.unigramcounts = defaultdict(int)
        self.bigramcounts = defaultdict(int)
        self.trigramcounts = defaultdict(int)
        
        for sentence in corpus:
            
            unigrams = get_ngrams(sentence, 1)
            for unigram in unigrams:
                self.unigramcounts[unigram] += 1
            
            bigrams = get_ngrams(sentence, 2)
            for bigram in bigrams:
                self.bigramcounts[bigram] += 1
            
            trigrams = get_ngrams(sentence, 3)
            for trigram in trigrams:
                self.trigramcounts[trigram] += 1
        
        self.bigramcounts[('START', 'START')] = self.num_sentence # add the [('START','START')] eqaul to total sentence to solve part 3


    def raw_trigram_probability(self,trigram):
        if trigram in self.trigramcounts:
            if trigram[0:2] in self.bigramcounts:
                return self.trigramcounts[trigram]/self.bigramcounts[(trigram[0:2])]
            else:
                return 1/len(self.lexicon)
        else:
            if trigram[0:2] in self.bigramcounts:
                return 0
            else:
                return 1/len(self.lexicon)
            

    def raw_bigram_probability(self, bigram):
        if bigram in self.bigramcounts:
            if bigram[0:1] in self.unigramcounts:
                return self.bigramcounts[bigram]/self.unigramcounts[(bigram[0:1])]
            else:
                return 1/len(self.lexicon)
        else:
            if bigram[0:1] in self.unigramcounts:
                return 0
            else:
                return 1/len(self.lexicon)

    
    def raw_unigram_probability(self, unigram):
        total_words = sum(self.unigramcounts.values())-self.unigramcounts[('START',)] # Total number of words except Start
        if unigram == ('START',):
            return 0
        elif unigram in self.unigramcounts:
            return self.unigramcounts[unigram] / total_words 
        else:
            return 1/len(self.lexicon)

        #hint: recomputing the denominator every time the method is called
        # can be slow! You might want to compute the total number of words once, 
        # store in the TrigramModel instance, and then re-use it.  
        

    def generate_sentence(self,t=20): 
        sentence = ['START', 'START']
        while len(sentence) < t + 2:  
            context = (sentence[-2], sentence[-1])
            candidates = [key[2] for key in self.trigramcounts.keys() if key[:2] == context]
            probabilities = [self.raw_trigram_probability((context[0], context[1], candidate)) for candidate in candidates]
            if probabilities:
                next_word = random.choices(candidates, weights=probabilities, k=1)[0]
                if next_word == 'STOP':
                    break
                sentence.append(next_word)
            else:
                break
        return sentence[2:]        
         

    def smoothed_trigram_probability(self, trigram):

        lambda1 = 1/3.0
        lambda2 = 1/3.0
        lambda3 = 1/3.0
        return lambda1*self.raw_trigram_probability(trigram)+\
                lambda2*self.raw_bigram_probability((trigram[1:3]))+\
                lambda3*self.raw_unigram_probability((trigram[2]),)
        
    def sentence_logprob(self, sentence):
        log_prob = 0
        for i in range(len(sentence)-2):
            log_prob += math.log2(self.smoothed_trigram_probability(get_ngrams(sentence, 3)[i]))
                
        return log_prob
        


    def perplexity(self, corpus):
        tokens = sum(self.unigramcounts.values())-self.unigramcounts[('START',)] # Total number of words except Start
        sum_prob = 0
        sum_tokens = 0
        for sentence in corpus:
            sum_prob += self.sentence_logprob(sentence)
            sentence_tokens = get_ngrams(sentence, 1)
            sum_tokens += (len(sentence_tokens)-1)
        
        return 2**(-sum_prob/sum_tokens)


def essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2):

        model1 = TrigramModel(training_file1)
        model2 = TrigramModel(training_file2)

        total = 0
        correct = 0       
 
        for f in os.listdir(testdir1):
            total += 1
            pp = model1.perplexity(corpus_reader(os.path.join(testdir1, f), model1.lexicon))
            pp2 = model2.perplexity(corpus_reader(os.path.join(testdir1, f), model2.lexicon))
            if pp < pp2:
                correct += 1
    
        for f in os.listdir(testdir2):
            total += 1
            pp = model2.perplexity(corpus_reader(os.path.join(testdir2, f), model2.lexicon))
            pp2 = model1.perplexity(corpus_reader(os.path.join(testdir2, f), model1.lexicon))
            if pp < pp2:
                correct += 1
        
        return correct/total





# In[20]:


if __name__ == "__main__":
    
    training_file = "C:/Users/ASUS/Desktop/Semester 2/4705 NLP/HW1/hw1_data/brown_train.txt"
    model = TrigramModel(training_file) 

    # put test code here...
    # or run the script from the command line with 
    # $ python -i trigram_model.py [corpus_file]
    # >>> 
    #
    # you can then call methods on the model instance in the interactive 
    # Python prompt. 
    
    # Testing Part I get_ngrams:
    print(get_ngrams(["natural","language","processing"],1))
    print(get_ngrams(["natural","language","processing"],2))
    print(get_ngrams(["natural","language","processing"],3))
    
    # Testing Part II counting n-grams:
    print(model.trigramcounts[('START','START','the')])
    print(model.bigramcounts[('START','the')])
    print(model.unigramcounts[('the',)])
    
    # Testing Part III raw n-gram:
    print(model.raw_unigram_probability(('the',)))
    print(model.raw_bigram_probability(('START', 'the')))
    print(model.raw_trigram_probability(('START', 'the', 'fulton')))
    
    # Testing Part III generating sentence:
    for i in range(5):
        print(model.generate_sentence())
        i = i+1
        
    # Testing Part IV smoothed probability:
    print(model.smoothed_trigram_probability(('START', 'START', 'the')))
    
    # Testing for Part V sentence probability:
    print(model.sentence_logprob(['something', 'hard', 'grazed', 'his', 'knuckles', '.']))

    
    # Testing perplexity: 
    testfile = "C:/Users/ASUS/Desktop/Semester 2/4705 NLP/HW1/hw1_data/brown_test.txt"
    dev_corpus = corpus_reader(testfile, model.lexicon)
    pp = model.perplexity(dev_corpus)
    print("testing preplexity is {:.5f}".format(pp))


    # Essay scoring experiment: 
    training_file1 = "C:/Users/ASUS/Desktop/Semester 2/4705 NLP/HW1/hw1_data/ets_toefl_data/train_high.txt"
    training_file2 = "C:/Users/ASUS/Desktop/Semester 2/4705 NLP/HW1/hw1_data/ets_toefl_data/train_low.txt"
    testdir1="C:/Users/ASUS/Desktop/Semester 2/4705 NLP/HW1/hw1_data/ets_toefl_data/test_high"
    testdir2="C:/Users/ASUS/Desktop/Semester 2/4705 NLP/HW1/hw1_data/ets_toefl_data/test_low"
    acc = essay_scoring_experiment(training_file1, training_file2, testdir1, testdir2)
    print("model accuracy is {:.5f}".format(acc))


