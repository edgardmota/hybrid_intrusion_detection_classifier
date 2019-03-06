#!/usr/bin/env python3
import sys, os

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/rna")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/hybrid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/knn")

from rna_module import RnaModuleHO
from preprocessor import Preprocessor
from config_processor import Config_Processor
from experiment import Experiment

#PREPROCESSADOR PARA ATRIBUTOS CATEGORICOS
preprocessor = Preprocessor()
preprocessor.setColumnsCategory(['protocol_type','service','flag'])

ho = False
if ((len(sys.argv) > 1) and (sys.argv[1] == '--with-ho')):
    ho = True
    c_file = sys.argv[2]
else:
    c_file = sys.argv[1]
cp = Config_Processor(c_file, ho)
kwargs = cp.get_kwargs()
kwargs['preprocessor'] = preprocessor
if ho:
    RnaModuleHO(**kwargs)
else:
    Experiment(**kwargs).run()
