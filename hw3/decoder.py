import sys
import copy

import numpy as np
import torch
import torch.nn.functional as F
from conll_reader import DependencyStructure, DependencyEdge, conll_reader
from extract_training_data import FeatureExtractor, State
from train_model import DependencyModel

class Parser(object):

    def __init__(self, extractor, modelfile):
        self.extractor = extractor

        # Create a new model and load the parameters
        self.model = DependencyModel(len(extractor.word_vocab), len(extractor.output_labels))
        self.model.load_state_dict(torch.load(modelfile))
        sys.stderr.write("Done loading model")

        # The following dictionary from indices to output actions will be useful
        self.output_labels = dict([(index, action) for (action, index) in extractor.output_labels.items()])

    def parse_sentence(self, words, pos):

        state = State(range(1,len(words)))
        state.stack.append(0)

        while state.buffer:
            feature = self.extractor.get_input_representation(words, pos, state)
            softmax_vector = self.model(torch.LongTensor([feature]))
            softmax_vector= softmax_vector.detach().cpu().numpy()
            indices = np.argsort(-softmax_vector, axis=1)[0]
            for i in indices:
                key, val = self.output_labels[i]
                if key == 'shift':
                    if len(state.buffer) == 1:
                        if len(state.stack) == 0:
                            state.shift()
                            break
                    elif len(state.buffer) > 1:
                        state.shift()
                        break
                elif key == 'left_arc':
                    if len(state.stack) > 0:
                        if val != 'root':
                            state.left_arc(val)
                            break
                elif key == 'right_arc':
                    if len(state.stack) != 0:
                        state.right_arc(val)
                        break

        result = DependencyStructure()
        for p,c,r in state.deps:
            result.add_deprel(DependencyEdge(c,words[c],pos[c],p, r))

        return result


if __name__ == "__main__":

    WORD_VOCAB_FILE = 'data/words.vocab'
    POS_VOCAB_FILE = 'data/pos.vocab'

    try:
        word_vocab_f = open(WORD_VOCAB_FILE,'r')
        pos_vocab_f = open(POS_VOCAB_FILE,'r')
    except FileNotFoundError:
        print("Could not find vocabulary files {} and {}".format(WORD_VOCAB_FILE, POS_VOCAB_FILE))
        sys.exit(1)

    extractor = FeatureExtractor(word_vocab_f, pos_vocab_f)
    parser = Parser(extractor, sys.argv[1])

    with open(sys.argv[2],'r') as in_file:
        for dtree in conll_reader(in_file):
            words = dtree.words()
            pos = dtree.pos()
            deps = parser.parse_sentence(words, pos)
            print(deps.print_conll())
            print()
