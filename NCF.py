import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
import pydot

from sklearn.model_selection import train_test_split
import keras
from keras.optimizers import Adam
from keras.utils.vis_utils import model_to_dot
from sklearn.metrics import mean_absolute_error
from keras.utils.vis_utils import model_to_dot


class NeuralCollaborativeFiltering:

	def __init__(self, n_users, n_movies, n_latent_factors, learning_rate, n_epochs):
		keras.utils.vis_utils.pydot = pydot
		self.n_users = n_users
		self.n_movies = n_movies
		self.n_latent_factors = n_latent_factors
		self.learning_rate = learning_rate
		self.n_epochs = n_epochs

	def build_model(self):
		movie_input = keras.layers.Input(shape=[1],name='Item')
		movie_embedding = keras.layers.Embedding(self.n_movies + 1, self.n_latent_factors, name='Movie-Embedding')(movie_input)
		movie_vec = keras.layers.Flatten(name='FlattenMovies')(movie_embedding)

		user_input = keras.layers.Input(shape=[1],name='User')
		user_vec = keras.layers.Flatten(name='FlattenUsers')(keras.layers.Embedding(self.n_users + 1, self.n_latent_factors,name='User-Embedding')(user_input))

		concat = keras.layers.concatenate([movie_vec, user_vec])
		concat_dropout = keras.layers.Dropout(0.2)(concat)
		dense = keras.layers.Dense(200,name='FullyConnected')(concat)
		dropout_1 = keras.layers.Dropout(0.2,name='Dropout')(dense)
		dense_2 = keras.layers.Dense(100,name='FullyConnected-1')(concat)
		dropout_2 = keras.layers.Dropout(0.2,name='Dropout')(dense_2)
		dense_3 = keras.layers.Dense(50,name='FullyConnected-2')(dense_2)
		dropout_3 = keras.layers.Dropout(0.2,name='Dropout')(dense_3)
		dense_4 = keras.layers.Dense(20,name='FullyConnected-3', activation='relu')(dense_3)

		resultMF = keras.layers.dot([movie_vec, user_vec], axes = 1)
		resultMLP = keras.layers.Dense(1, activation='relu',name='Activation')(dense_4)
		combine_mlp_mf = keras.layers.average([resultMF, resultMLP])

		self.model = keras.Model([user_input, movie_input], combine_mlp_mf)
		adam = Adam(lr=self.learning_rate)
		self.model.compile(optimizer=adam,loss= 'mean_absolute_error')

	def fit(self, user_item_list, rating):
		self.history = self.model.fit(user_item_list, rating, epochs=self.n_epochs, verbose = 1)
		return self.history

	def predict(self, user_item_list):
		return np.round(self.model.predict(user_item_list),0)