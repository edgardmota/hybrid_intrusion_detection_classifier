#!/usr/bin/env python3
import sys, os
import pandas as pd
from preprocessor import Preprocessor
from dataSet import DataSet
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/rna")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/hybrid")
sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/knn")

from cross_validation import CrossValidation
from preprocessor import Preprocessor
from dataSet import DataSet
from knn_classifier import KnnClassifier
from rna_classifier import RnaClassifier
from hybrid_classifier import HybridClassifier
from rna_module import RnaModule, RnaModuleHO
from knn_module import KnnModule
from evaluate_module import EvaluateModule
from keras.optimizers import Adam, Nadam, RMSprop
from keras.activations import tanh, sigmoid, relu, elu
from keras.losses import logcosh, binary_crossentropy

dts = DataSet()
dts.setFilePath("bases/sub_bases_nslkdd_20attribute/")
#dts.setFileName("base_iris.csv")
#dts.setFileName("SmallTrainingSet.csv")
##dts.setFileName("winequality-red.csv")
#dts.setFileName("NSL_KDD-master/20PercentTrainingSet.csv")
dts.setFileName("NSL_KDD-master/KDDTrain+binary_class.csv")
#dts.setFileName("NSL_KDD-master/SmallTrainingSet.csv")
#dts.setFileName("NSL_KDD-master/SmallTrainingSetFiveClass.csv")

#dts.setFileName("../../KDDCUP99/kddcup10%.csv")


#print("load data")
#ts.loadData(10)


#CONFIGURACAO DO KNN
knn = KnnModule()
knn.setKNeighbors(1)
knn_classifier = KnnClassifier()
knn_classifier.setKnn(knn)

#PREPROCESSADOR PARA ATRIBUTOS CATEGORICOS
preprocessor = Preprocessor()
preprocessor.setColumnsCategory(['protocol_type','service','flag'])

if ((len(sys.argv) > 1) and (sys.argv[1] == '--with-ho')):
    with_ho = True

    hc = {
        'lr': (0.5, 5, 10),
        'first_neuron': [4, 8, 16, 32, 41, 64],
        'dropout': (0, 0.5, 5),
        'hidden_layers': [0, 1, 2],
        'losses': [logcosh, binary_crossentropy],
        'optimizer': [Adam, Nadam, RMSprop],
        'activation': [tanh, sigmoid, relu, elu],
        'last_activation': [tanh, sigmoid, relu, elu]
    }

    # hc = {
    #     'n_neurons_input': [41],
    #     'input_dim_neurons': [41],
    #     'n_neurons_hidden': [41],
    #     'n_neurons_output': [1],
    #     'i_activation_function': ['tanh', 'sigmoid', 'relu', 'elu'],
    #     'h_activation_function': ['tanh', 'sigmoid', 'relu', 'elu'],
    #     'o_activation_function': ['tanh', 'sigmoid', 'relu', 'elu'],
    # }
    ds = DataSet().loadSubDataSet('bases/sub_bases_small_training_set1000/full_data_set.csv')
    ho_results_file = 'ho_results'
    rna_ho = RnaModuleHO(ds,preprocessor,hc,ho_results_file)
    best_hyperparameters = rna_ho.getHyperparameters()

    h = {
        'n_neurons_input': best_hyperparameters['n_neurons_input'],
        'input_dim_neurons': best_hyperparameters['input_dim_neurons'],
        'n_neurons_hidden': best_hyperparameters['n_neurons_hidden'],
        'n_neurons_output': best_hyperparameters['n_neurons_output'],
        'i_activation_function': best_hyperparameters['i_activation_function'],
        'h_activation_function': best_hyperparameters['h_activation_function'],
        'o_activation_function': best_hyperparameters['o_activation_function'],
    }
else:
    with_ho = False
    # h = {
    #     'n_neurons_input': 41,
    #     'input_dim_neurons': 41,
    #     'n_neurons_hidden': 41,
    #     'n_neurons_output': 1,
    #     'i_activation_function': 'tanh',
    #     'h_activation_function': 'tanh',
    #     'o_activation_function': 'tanh',
    # }

    h = {
        'n_neurons_input': 32,
        'input_dim_neurons': 41,
        'n_neurons_hidden': 32,
        'n_neurons_output': 1,
        'i_activation_function': 'sigmoid',
        'h_activation_function': 'sigmoid',
        'o_activation_function': 'tanh',
    }

#CONFIGURACAO DA REDE NEURAL
rna = RnaModule()

rna.setNumberNeuronsInputLayer(h['n_neurons_input'])
rna.setActivationFunctionInputLayer(h['i_activation_function'])
rna.setInputDimNeurons(h['input_dim_neurons'])
rna.setNumberNeuronsHiddenLayer(h['n_neurons_hidden'])
rna.setActivationFunctionHiddenLayer(h['h_activation_function'])
rna.setNumberNeuronsOutputLayer(h['n_neurons_output'])
rna.setActivationFunctionOutputLayer(h['o_activation_function'])

rna_classifier = RnaClassifier()
rna_classifier.setRna(rna)

#METODO HIBRIDO
hybrid_classifier = HybridClassifier()
hybrid_classifier.setPercentilFaixaSup(25)
hybrid_classifier.setPercentilFaixaInf(100)
hybrid_classifier.setRna(rna)
hybrid_classifier.setKnn(knn)

evaluate = EvaluateModule()

cross = CrossValidation()
#DEFINIR A ITERACAO QUE O CROSS VALIDATION ESTA
cross.setIteration(1)
cross.setK(10)
cross.setPreprocessor(preprocessor)
#cross.setFilePath("bases/sub_bases_20_nslkdd/")
#cross.setFilePath("bases/sub_bases_train+_nslkdd/")
#cross.setFilePath("bases/sub_bases_nslkdd_tcp_attribute/")
#cross.setFilePath("bases/sub_bases_nslkdd_12attribute/")
#cross.setFilePath("bases/sub_bases_nslkdd_20attribute/")
#cross.setFilePath("bases/sub_bases_SmallTrainingSet/")
cross.setFilePath("bases/sub_bases_small_training_set1000/")

#cross.setResultPath("results/faixa_hibrido/")
if with_ho:
    cross.setResultPath("results/with_ho/")
else:
    cross.setResultPath("results/teste_casa/")

#cross.setClassifier(rna_classifier)
#cross.setClassifier(knn_classifier)
#cross.setClassifier(clustered_knn_classifier)
#cross.setClassifier(clustered_density_knn_classifier)
cross.setClassifier(hybrid_classifier)

cross.setEvaluateModule(evaluate)
cross.run()
