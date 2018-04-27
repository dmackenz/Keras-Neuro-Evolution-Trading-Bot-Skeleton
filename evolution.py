from keras.models import Sequential
from keras.layers.core import Dense
import Population

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
    pop_size = 50
    mutation_rate = 0.05
    mutation_scale = 0.3
    starting_cash = 1
    trading_fee = 0

    prices = [
                4,
                3,
                2,
                1,
                5,
                6
            ]

    inputs = [
                [-0.2, 0.2, 0.4, -0.5],
                [-0.2, 0.1, 0.4, -0.1],
                [-0.2, 0.2, 0.4, -0.2],
                [-0.2, 0.3, 0.3, -0.3],
                [0.4, -0.9, -0.3, 0.1],
                [0.8, -0.6, -0.5, 0.1]
            ]

    pop = Population.Population(pop_size, build_model, mutation_rate, mutation_scale, starting_cash, prices[0], trading_fee)

    while True:
    # for i in range(1):
        pop.evolve(inputs, prices)