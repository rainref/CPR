
from __future__ import unicode_literals
import unittest
import codecs
import re


import os,sys,inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)

from subword_nmt import apply_bpe

class BPEUtil:
    def __init__(self,path):
        with codecs.open(path, encoding='utf-8') as bpefile:
            self.bpe = apply_bpe.BPE(bpefile)


if __name__ == '__main__':
    bpeTool = BPEUtil(os.path.join(currentdir,  'bpe_vocab.out'))
    bpeTool.bpe.process_lines(filename=os.path.join(currentdir,  'output_code.txt'),outfile=os.path.join(currentdir,  'output_code.txt.bpe'))
