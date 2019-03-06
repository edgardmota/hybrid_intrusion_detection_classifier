#!/usr/bin/env python3
import configparser
from keras.optimizers import Adam, Nadam, RMSprop
from keras.activations import tanh, sigmoid, relu, elu
from keras.losses import logcosh, binary_crossentropy
from dataSet import DataSet

class Config_Processor(object):

    def __init__(self, file, ho=False):
        self.cp = configparser.ConfigParser(delimiters=['='])
        self.ho = ho
        self.cp.read(file)

    def _kwargs_setting(self, kwargs, section, key, value):
        if section == 'hyperparameters' and self.ho:
            try:
                kwargs['hc'][key] = value
            except:
                kwargs['hc'] = dict()
                kwargs['hc'][key] = value
        else:
            kwargs[key] = value

    def get_kwargs(self):
        MAPPINGS_HO = {
            'hyperparameters': lambda v: eval(v),
            'general': [
                (['output_file_prefix', 'exp_no'], None),
                (['ho_dataset'], lambda v: DataSet().loadSubDataSet(v))
            ],
        }

        MAPPINGS = {
            'general':[
                (['dataset_path', 'results_path'], None),
                (['iteration', 'percentile_upper_range', 'percentile_bottom_range'], lambda v: int(v)),
            ],
            'report':[
                (['report_file', 'metric', 'max_min', 'transformation_suffix'], None),
                (['columns'], lambda v: eval(v)),
                (['transformations'], lambda v: list(filter(lambda x: len(x) > 0, map(lambda x: x.split(), v.splitlines())))),
            ],
            'hyperparameters': [
                (['input_dim_neurons', 'dropout', 'hidden_layers', 'neighbors', 'k', 'n_neurons_hidden', 'n_neurons_output', 'n_neurons_input', ], lambda v: int(v)),
                (['i_activation_function', 'h_activation_function', 'o_activation_function', 'losses'], None),
                (['optimizer'], lambda v: eval(v)),
                (['lr'], lambda v: float(v)),
            ],
        }

        kwargs = dict()
        if self.ho:
            mappings = MAPPINGS_HO
        else:
            mappings = MAPPINGS
        for section in mappings.keys():
            try:
                if callable(mappings[section]):
                    for key in list(self.cp[section].keys()):
                        self._kwargs_setting(kwargs, section, key, mappings[section](self.cp[section][key]))
                else:
                    for mapping in mappings[section]:
                        if mapping[1]:
                            if ('transformations' in mapping[0]) and (len(mapping[0]) == 1):
                                processed_list = []
                                unproccessed_list = mapping[1](self.cp[section][mapping[0][0]])
                                for item in unproccessed_list:
                                    processed_list.append(tuple([item[0], eval("lambda {}: {}".format(item[1], item[2]))]))
                                kwargs['transformations'] = processed_list
                            else:
                                for key in mapping[0]:
                                    try:
                                        self._kwargs_setting(kwargs, section, key, mapping[1](self.cp[section][key]))
                                    except KeyError:
                                        pass
                        else:
                            for key in mapping[0]:
                                try:
                                    self._kwargs_setting(kwargs, section, key, self.cp[section][key])
                                except KeyError:
                                    pass
            except KeyError:
                pass
        return kwargs
