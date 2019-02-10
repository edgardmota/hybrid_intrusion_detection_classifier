import talos as ta
from talos import Reporting
from talos.model.layers import hidden_layers
from talos.model.normalizers import lr_normalizer
import numpy as np
import tensorflow as tf
from keras.callbacks import CSVLogger
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
import keras.preprocessing.text
from keras.preprocessing import sequence
from keras import backend as K
from keras.callbacks import EarlyStopping
from keras.optimizers import Adam, Nadam, RMSprop

class RnaModule(object):
	#conjuto de exemplos de treino
	data_set_samples = []
	#classes dos exemplos de treino
	data_set_labels = []
	#conjunto de exemplos de teste
	test_data_set_samples = []
	#classes dos exemplos de teste
	test_data_set_labels = []

	number_neurons_input_layer = 0
	number_neurons_hidden_layer = 0
	number_neurons_output_layer = 0

	input_dim_neurons = 0

	#funcoes de ativacao dos neuronios de cada camada
	activation_function_input_layer = "relu"
	activation_function_hidden_layer = "relu"
	activation_function_output_layer = "sigmoid"

	model = None

	def __init__(self):
		print("init rna module")

	#funcao para criar a rna para abordagem simples
	def generateModel(self, inner_call=False):
		self.model = Sequential()
		self.model.add(Dense(self.number_neurons_input_layer, input_dim= self.input_dim_neurons, init='normal', activation=self.activation_function_input_layer))
		self.model.add(Dense(self.number_neurons_hidden_layer, init='normal', activation=self.activation_function_hidden_layer))
		self.model.add(Dense(self.number_neurons_output_layer, init='normal', activation=self.activation_function_output_layer))

		# self.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		self.model.compile(loss='logcosh', optimizer=RMSprop(lr=lr_normalizer(2.75,RMSprop)), metrics=['accuracy'])
		csv_logger = CSVLogger('training.log')
		#funcao para interromper treinamento quando o erro for suficientemente pequeno
		early_stopping = EarlyStopping(monitor='loss', patience=20)

		fit = self.model.fit(self.data_set_samples, self.data_set_labels, nb_epoch=500, verbose=2, callbacks=[early_stopping])
		if inner_call:
			return fit

    #funcao para criar a rna para a abordagem hibrida
	def generateHybridModel(self):
		# self.model = Sequential()
		# self.model.add(Dense(self.number_neurons_input_layer, input_dim= self.input_dim_neurons, init='normal', activation=self.activation_function_input_layer))
		# self.model.add(Dense(self.number_neurons_hidden_layer, init='normal', activation=self.activation_function_hidden_layer))
		# self.model.add(Dense(self.number_neurons_hidden_layer, init='normal', activation=self.activation_function_hidden_layer))
		# self.model.add(Dense(self.number_neurons_output_layer, init='normal', activation=self.activation_function_output_layer))
		#
		# self.model.compile(loss='logcosh', optimizer=RMSprop(lr=lr_normalizer(4.1,RMSprop)), metrics=['accuracy'])
		# csv_logger = CSVLogger('training.log')
		# #funcao para interromper treinamento quando o erro for suficientemente pequeno
		# early_stopping = EarlyStopping(monitor='loss', patience=20)
		#
		# fit = self.model.fit(self.data_set_samples, self.data_set_labels, nb_epoch=500, verbose=2, callbacks=[early_stopping])
		fit = self.generateModel(True)

		#obter valores da camada de saida da ultima iteracao do treinamento
		get_3rd_layer_output = K.function([self.model.layers[0].input], [self.model.layers[2].output])
		layer_output = get_3rd_layer_output([self.data_set_samples])[0]


		predictions = self.model.predict_classes(self.data_set_samples)

		return layer_output, predictions, fit

	#funcao utilizada para retornar o resultado da classificacao em termos de -1 a 1 (utilizada para a abordagem hibrida)
	def predict(self):
		predictions = self.model.predict(self.test_data_set_samples)
		return predictions

	#funcao utilizada para retornar o resultado da classificacao em 1 ou 0(utilizada para a abordagem simples)
	def predictClasses(self):
		predictions = self.model.predict_classes(self.test_data_set_samples)
		return predictions

	def setDataSet(self, data_set):
		self.data_set_samples = data_set.values[:,0:(len(data_set.values[0])-2)]
		self.data_set_labels = data_set.values[:,(len(data_set.values[0])-2)]
		#print(self.data_set_samples)
		#print(self.data_set_labels)

	def setTestDataSet(self, test_data_set):
		self.test_data_set_samples = test_data_set.values[:,0:(len(test_data_set.values[0])-2)]
		self.test_data_set_labels = test_data_set.values[:,(len(test_data_set.values[0])-2)]
		#print(self.test_data_set_samples)
		#print(self.test_data_set_labels)

	def setNumberNeuronsInputLayer(self, number):
		self.number_neurons_input_layer = number

	def getNumberNeuronsInputLayer(self):
		return self.number_neurons_input_layer

	def setNumberNeuronsHiddenLayer(self, number):
		self.number_neurons_hidden_layer = number

	def getNumberNeuronsHiddenLayer(self):
		return self.number_neurons_hidden_layer

	def setNumberNeuronsOutputLayer(self, number):
		self.number_neurons_output_layer = number

	def getNumberNeuronsOutputLayer(self):
		return self.number_neurons_output_layer

	def setActivationFunctionInputLayer(self, activation_function):
		self.activation_function_input_layer = activation_function

	def getActivationFunctionInputLayer(self):
		return self.activation_function_input_layer

	def setActivationFunctionHiddenLayer(self, activation_function):
		self.activation_function_hidden_layer = activation_function

	def getActivationFunctionHiddenLayer(self):
		return self.activation_function_hidden_layer

	def setActivationFunctionOutputLayer(self, activation_function):
		self.activation_function_output_layer = activation_function

	def getActivationFunctionOutputLayer(self):
		return self.activation_function_output_layer

	def setInputDimNeurons(self, number):
		self.input_dim_neurons = number

	def getNumberNeuronsInputLayer(self):
		return self.input_dim_neurons

	def setDimInputLayer(self, dim_input_layer):
		self.dim_input_layer = dim_input_layer

	def getDimInputLayer(self):
		return self.dim_input_layer

class RnaModuleHO:

	def __init__ (self, ds, pp, hc, ho_results_file):
		self.ho_results_file = ho_results_file + '_.csv'
		self.pp = pp
		self.pp.setDataSet(ds)
		self.pp.setTestDataSet(ds) #Gambiarra pois exige dois datasets
		ds, _ = self.pp.transformCategory()
		self.x = ds.values[:,0:(len(ds.values[0])-2)]
		self.y = ds.values[:,(len(ds.values[0])-2)]
		self.hc = hc
		self.rna = RnaModule()
		self._doHO()

	def __generateModelHO(self,x_train, y_train, x_val, y_val, params):
		self.rna.model = Sequential()
		self.rna.model.add(Dense(params['first_neuron'], input_dim=x_train.shape[1], activation=params['activation']))
		# if we want to also test for number of layers and shapes, that's possible
		self.rna.model.add(Dropout(params['dropout']))
		hidden_layers(self.rna.model, params, 1)
		# then we finish again with completely standard Keras way
		self.rna.model.add(Dense(1, init='normal', activation=params['last_activation']))
		self.rna.model.compile(loss=params['losses'],
			optimizer=params['optimizer'](lr=lr_normalizer(params['lr'],params['optimizer'])),
			metrics=['accuracy'])

		early_stopping = EarlyStopping(monitor='loss',patience=20)
		out = self.rna.model.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=500, verbose=0, callbacks=[early_stopping])

		return out, self.rna.model

    	# self.rna.model = Sequential()
		# self.rna.model.add(Dense(params['n_neurons_input'], input_dim=params['input_dim_neurons'], init='normal', activation=params['i_activation_function']))
		# self.rna.model.add(Dense(params['n_neurons_hidden'], init='normal', activation=params['h_activation_function']))
		# self.rna.model.add(Dense(params['n_neurons_output'], init='normal', activation=params['o_activation_function']))
		#
		# self.rna.model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
		#
		# #funcao para interromper treinamento quando o erro for suficientemente pequeno
		# early_stopping = EarlyStopping(monitor='loss',patience=20)
		#
		# out = self.rna.model.fit(x_train, y_train, validation_data=[x_val, y_val], epochs=500, verbose=2, callbacks=[early_stopping])
		#
		# return out, self.rna.model

	#Faz a otimização de hiperparâmetros
	def _doHO(self):
		self.scan_object = ta.Scan(self.x, self.y, self.hc, self.__generateModelHO, dataset_name=self.ho_results_file[:-5])

	def getHyperparameters(self):
		r = Reporting(self.ho_results_file)
		print(r.high())
		return r.best_params()
