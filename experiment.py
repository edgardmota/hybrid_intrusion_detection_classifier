#!/usr/bin/env python3
import pandas
from knn_classifier import KnnClassifier
from rna_classifier import RnaClassifier
from hybrid_classifier import HybridClassifier
from rna_module import RnaModule
from knn_module import KnnModule
from evaluate_module import EvaluateModule
from cross_validation import CrossValidation
from talos import Reporting

class Experiment(object):
    exp = 0

    def __init__(self, **kwargs):
        h = None
        try:
            kwargs_report = dict((k, kwargs[k]) for k in ('report_file', 'metric', 'max_min', 'columns', 'transformations'))
            h = self._load_h_from_register(**kwargs_report)
        except KeyError:
            pass
        if h:
            self.h_list = []
            for setting in h:
                self.h_list.append({**kwargs, **setting})
        else:
            self.h_list = [kwargs]

    def _run(self):
        if self.exp in range(len(self.h_list)):
            #CONFIGURACAO DO KNN
            knn = KnnModule()
            knn.setKNeighbors(self.h_list[self.exp]['neighbors'])
            knn_classifier = KnnClassifier()
            knn_classifier.setKnn(knn)

            rna = RnaModule()

            rna.setNumberNeuronsInputLayer(self.h_list[self.exp]['n_neurons_input'])
            rna.setActivationFunctionInputLayer(self.h_list[self.exp]['i_activation_function'])
            rna.setDropout(self.h_list[self.exp]['dropout'])
            rna.setNumberHiddenLayers(self.h_list[self.exp]['hidden_layers'])
            rna.setNumberNeuronsHiddenLayer(self.h_list[self.exp]['n_neurons_hidden'])
            rna.setActivationFunctionHiddenLayer(self.h_list[self.exp]['h_activation_function'])
            rna.setNumberNeuronsOutputLayer(self.h_list[self.exp]['n_neurons_output'])
            rna.setActivationFunctionOutputLayer(self.h_list[self.exp]['o_activation_function'])
            rna.setLossFunction(self.h_list[self.exp]['losses'])
            rna.setOptimizer(self.h_list[self.exp]['optimizer'])
            try:
                rna.setLearningRate(self.h_list[self.exp]['lr'])
            except:
                pass

            rna_classifier = RnaClassifier()
            rna_classifier.setRna(rna)

            #METODO HIBRIDO
            hybrid_classifier = HybridClassifier()
            hybrid_classifier.setPercentilFaixaSup(self.h_list[self.exp]['percentile_upper_range'])
            hybrid_classifier.setPercentilFaixaInf(self.h_list[self.exp]['percentile_bottom_range'])
            hybrid_classifier.setRna(rna)
            hybrid_classifier.setKnn(knn)

            evaluate = EvaluateModule(self.h_list[self.exp])

            self.cross = CrossValidation()

            #DEFINIR A ITERACAO QUE O CROSS VALIDATION ESTA
            self.cross.setIteration(self.h_list[self.exp]['iteration'])
            self.cross.setK(self.h_list[self.exp]['k'])
            self.cross.setPreprocessor(self.h_list[self.exp]['preprocessor'])

            self.cross.setFilePath(self.h_list[self.exp]['dataset_path'])
            if len(self.h_list) > 1:
                self.cross.setResultPath("{}/{}-{}/".format(self.h_list[self.exp]['results_path'], self.exp, len(self.h_list) - 1))
            else:
                self.cross.setResultPath(self.h_list[self.exp]['results_path'])
            self.cross.setClassifier(hybrid_classifier)

            self.cross.setEvaluateModule(evaluate)
            self.cross.run()
            del evaluate, knn, rna, rna_classifier, hybrid_classifier, self.cross
            return True
        else:
            return False

    def run(self):
        while self._run():
            self.exp += 1

    def _load_h_from_register(self, **kwargs):
        r =  Reporting(kwargs['report_file'])
        if kwargs['max_min'] == 'max':
            interest_value = r.high(kwargs['metric'])
        elif kwargs['max_min'] == 'min':
            interest_value = r.low(kwargs['metric'])
        else:
            raise(Exception)

        for transformation in kwargs['transformations']:
            r.data[transformation[0]] = r.data.loc[:,transformation[0]].apply(transformation[1])
        df = r.data
        new_df = pandas.DataFrame()
        cs = []
        for c in kwargs['columns']:
            if isinstance(c,dict):
                key = list(c.keys())[0]
                for derived_name in c[key]:
                    new_df[derived_name] = df[key]
                    cs.append(derived_name)
            else:
                new_df[c] = df[c]
                cs.append(c)
            new_df[kwargs['metric']] = df[kwargs['metric']]
        return new_df.loc[(new_df[kwargs['metric']] == interest_value), cs].to_dict('records')
