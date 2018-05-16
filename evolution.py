from keras.models import Sequential
from keras.layers.core import Dense
from utils.Population import Population
import numpy as np
from time import time


def build_model():
    num_inputs = 4
    hidden_nodes = 16
    num_outputs = 3

    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=num_inputs))
    model.add(Dense(num_outputs, activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    
    return model

if __name__ == '__main__':
    # suppress tf GPU logging
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    pop_size = 50
    mutation_rate = 0.05
    mutation_scale = 0.3
    starting_cash = 1.
    trading_fee = 0
    generations = 10

    # generate random test data
    test_size = 100
    np.random.seed(42)
    prices = np.random.normal(10, 0.1, test_size)
    inputs = np.random.rand(test_size, 4) * 2 - 1

    # build initial population
    pop = Population(pop_size, build_model, mutation_rate, 
                     mutation_scale, starting_cash, prices[0], trading_fee)

    # run defined number of evolutions
    for i in range(generations):
        start = time()
        pop.evolve(inputs, prices)
        print('\n\nDuration: {0:.2f}s'.format(time()-start))
