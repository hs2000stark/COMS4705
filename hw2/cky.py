"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Probabilistic Context Free Grammars 
Daniel Bauer
"""
import math
import sys
from collections import defaultdict
import itertools
from grammar import Pcfg

### Use the following two functions to check the format of your data structures in part 3 ###
def check_table_format(table):
    """
    Return true if the backpointer table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Backpointer table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and \
          isinstance(split[0], int)  and isinstance(split[1], int):
            sys.stderr.write("Keys of the backpointer table must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of backpointer table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            bps = table[split][nt]
            if isinstance(bps, str): # Leaf nodes may be strings
                continue 
            if not isinstance(bps, tuple):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Incorrect type: {}\n".format(bps))
                return False
            if len(bps) != 2:
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Found more than two backpointers: {}\n".format(bps))
                return False
            for bp in bps: 
                if not isinstance(bp, tuple) or len(bp)!=3:
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has length != 3.\n".format(bp))
                    return False
                if not (isinstance(bp[0], str) and isinstance(bp[1], int) and isinstance(bp[2], int)):
                    print(bp)
                    sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a pair ((i,k,A),(k,j,B)) of backpointers. Backpointer has incorrect type.\n".format(bp))
                    return False
    return True

def check_probs_format(table):
    """
    Return true if the probability table object is formatted correctly.
    Otherwise return False and print an error.  
    """
    if not isinstance(table, dict): 
        sys.stderr.write("Probability table is not a dict.\n")
        return False
    for split in table: 
        if not isinstance(split, tuple) and len(split) ==2 and isinstance(split[0], int) and isinstance(split[1], int):
            sys.stderr.write("Keys of the probability must be tuples (i,j) representing spans.\n")
            return False
        if not isinstance(table[split], dict):
            sys.stderr.write("Value of probability table (for each span) is not a dict.\n")
            return False
        for nt in table[split]:
            if not isinstance(nt, str): 
                sys.stderr.write("Keys of the inner dictionary (for each span) must be strings representing nonterminals.\n")
                return False
            prob = table[split][nt]
            if not isinstance(prob, float):
                sys.stderr.write("Values of the inner dictionary (for each span and nonterminal) must be a float.{}\n".format(prob))
                return False
            if prob > 0:
                sys.stderr.write("Log probability may not be > 0.  {}\n".format(prob))
                return False
    return True



class CkyParser(object):
    """
    A CKY parser.
    """

    def __init__(self, grammar): 
        """
        Initialize a new parser instance from a grammar. 
        """
        self.grammar = grammar

    def is_in_language(self, tokens):
        table = {}
        n = len(tokens)
        ### initialize the CKY table
        for i, token in enumerate(tokens):
            for rhs, rules in self.grammar.rhs_to_rules.items():
                if (token) in rhs:
                    for rule in rules:
                        if (i, i + 1) not in table:
                            table[(i, i + 1)] = {}
                        table[(i, i + 1)].update({rule[0]: token})
        ### implement the CKY algorithm
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    if (i, k) in table and (k, j) in table:
                        for B in table[(i, k)]:
                            for C in table[(k, j)]:
                                if (B, C) in self.grammar.rhs_to_rules:
                                    for rule in self.grammar.rhs_to_rules[(B, C)]:
                                        if (i, j) not in table:
                                            table[(i, j)] = {}
                                        table[(i, j)].update({rule[0]: ((B, i, k), (C, k, j))})

        return (0, n) in table and self.grammar.startsymbol in table[(0, n)]
       
    def parse_with_backpointers(self, tokens):
        table = defaultdict(dict)
        probs = defaultdict(dict)

        for i in range(len(tokens)):
            for j in self.grammar.rhs_to_rules[(tokens[i]),]:
                table[(i, i + 1)][j[0]] = j[1][0]
                probs[(i, i + 1)][j[0]] = math.log(j[2])

        for length in range(2, len(tokens) + 1):
            for i in range(len(tokens) - length + 1):
                j = i + length
                for k in range(i + 1, j):
                    for key1, val1 in probs[(i, k)].items():
                        for key2, val2 in probs[(k, j)].items():
                            for z in self.grammar.rhs_to_rules[(key1, key2)]:
                                if not probs[(i, j)].get(z[0]):
                                    probs[(i, j)][z[0]] = math.log(z[2]) + probs[(i, k)][key1] + probs[(k, j)][key2]
                                    table[(i, j)][z[0]] = ((key1, i, k), (key2, k, j))
                                else:
                                    res = math.log(z[2]) + probs[(i, k)][key1] + probs[(k, j)][key2]
                                    if res > probs[(i, j)][z[0]]:
                                        probs[(i, j)][z[0]] = res
                                        table[(i, j)][z[0]] = ((key1, i, k), (key2, k, j))

        return table, probs


def get_tree(chart, i,j,nt):

    if nt not in chart[(i, j)]:
        raise KeyError


    if j - i == 1:
        return (nt, chart[(i, j)][nt])

    else:
        res1 = get_tree(chart, chart[(i, j)][nt][0][1], chart[(i, j)][nt][0][2], chart[(i, j)][nt][0][0])
        res2 = get_tree(chart, chart[(i, j)][nt][1][1], chart[(i, j)][nt][1][2], chart[(i, j)][nt][1][0])

    return (nt, res1, res2)
 
       
if __name__ == "__main__":
    
    with open('atis3.pcfg','r') as grammar_file: 
        grammar = Pcfg(grammar_file) 
        parser = CkyParser(grammar)
        toks =['flights', 'from','miami', 'to', 'cleveland','.'] 
        print(parser.is_in_language(toks))
        toks = ['miami', 'flights', 'cleveland', 'from', 'to', '.']
        print(parser.is_in_language(toks))
        print(toks)
        #table,probs = parser.parse_with_backpointers(toks)
        #assert check_table_format(chart)
        #assert check_probs_format(probs)
        
