from keras.models import Sequential
from keras.layers.core import Dense
import Population

def build_model():
    num_inputs = 2
    hidden_nodes = 16
    num_outputs = 3

    model = Sequential()
    model.add(Dense(hidden_nodes, activation='relu', input_dim=num_inputs))
    model.add(Dense(num_outputs, activation='softmax'))
    model.compile(loss='mse', optimizer='adam')
    return model

if __name__ == '__main__':
    pop_size = 10
    mutation_rate = 0.1
    starting_cash = 1
    trading_fee = 0

    prices = [
                1,
                2,
                3,
                4,
                5,
                6
            ]

    inputs = [
                [2, 6],
                [9, 1],
                [1, 7],
                [7, 3],
                [4, 9],
                [8, 2]
            ]

    pop = Population.Population(pop_size, build_model, mutation_rate, starting_cash, prices[0], trading_fee)

    while True:
        pop.evolve(inputs, prices)