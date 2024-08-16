"""
COMS W4705 - Natural Language Processing - Fall 2023
Homework 2 - Parsing with Context Free Grammars 
Daniel Bauer
"""

import sys
from collections import defaultdict
from math import fsum


class Pcfg(object):
    """
    Represent a probabilistic context free grammar. 
    """

    def __init__(self, grammar_file): 
        self.rhs_to_rules = defaultdict(list)
        self.lhs_to_rules = defaultdict(list)
        self.startsymbol = None 
        self.read_rules(grammar_file)      
 
    def read_rules(self,grammar_file):
        
        for line in grammar_file: 
            line = line.strip()
            if line and not line.startswith("#"):
                if "->" in line: 
                    rule = self.parse_rule(line.strip())
                    lhs, rhs, prob = rule
                    self.rhs_to_rules[rhs].append(rule)
                    self.lhs_to_rules[lhs].append(rule)
                else: 
                    startsymbol, prob = line.rsplit(";")
                    self.startsymbol = startsymbol.strip()
                    
     
    def parse_rule(self,rule_s):
        lhs, other = rule_s.split("->")
        lhs = lhs.strip()
        rhs_s, prob_s = other.rsplit(";",1) 
        prob = float(prob_s)
        rhs = tuple(rhs_s.strip().split())
        return (lhs, rhs, prob)

    def verify_grammar(self):
        """
        Return True if the grammar is a valid PCFG in CNF.
        Otherwise return False. 
        """
        for lhs, rules in self.lhs_to_rules.items():
            prob_sum = 0.0
            for rule in rules:
                _, rhs, prob = rule
                # Sum probabilities for the current lhs
                prob_sum += prob

                # Check if rule is in CNF
                if len(rhs) == 2:  # For A -> BC, check if B and C are non-terminals
                    if not all(item in self.lhs_to_rules for item in rhs):
                        return False
                elif len(rhs) == 1:  # For A -> a, check if a is not a non-terminal
                    if rhs[0] in self.lhs_to_rules:
                        return False
                else:  # Any other form is not allowed in CNF
                    return False

            # Check if probabilities for the current lhs sum to approximately 1.0
            if not isclose(prob_sum, 1.0, rel_tol=1e-9):
                return False

        return True


if __name__ == "__main__":
    with open("C:/Users/ASUS/Desktop/Semester 2/4705 NLP/data/hw2/atis3.pcfg",'r') as grammar_file:
        grammar = Pcfg(grammar_file)
        
