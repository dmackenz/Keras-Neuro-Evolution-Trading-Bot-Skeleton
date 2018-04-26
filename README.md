# Keras-Neuro-Evolution-Trading-Bot-Skeleton
This project outlines the skeleton for creating a neuro-evolution trading bot with a Keras neural network. This is part of a reinforcement learning strategy to "reward" the neural network whenever it creates a trading strategy that generates profit. After each generation of neural networks, the highest profit networks will carry on to the next generation and produce a new generation of networks that are slightly mutated based on those with the highest profits. This mutation will create new trading strategies based on the inputs provided. If the mutation is not aligned with the end goal of generating profit, then it most likely will not carry on to the next generation.

This is in early stages of development and has only proven that it is capable of evolving a neural network to generate profit in a trading bot scenario. This project does not directly connect to an exchange API (but could be easily extendable via the Wallet class). I am not responsible for other's use cases of this program and the money that they make or lose with this program.

# Fitness Normalization Algorithm (pseudo-code)
NOTE: fitness is the percentage of profit that the bot earned from its starting capital
```Python
# get min and max fitness
min_fitness = min(fitnesses)
max_fitness = max(fitnesses)

# linearly scale all fitness between 0 & 1 then square giving higher fitness more of an edge
for fitness in fitnesses:
	fitness = ((fitness - min_fitness) - (max_fitness - min_fitness)) ** 2

# get sum of new fitnesses
sum_fitnesss = sum(fitneses)

# bound new fitness between 0 and 1 
for fitness in fitnesses:
	fitness /= sum_fitness
```

# Selection Algorithm (pseudo-code)
```Python
cnt = 0
index = 0

# sort fitnesses in descending order
fitnesses.sort(reversed=True)

# generate random number between 0 and 1
r = np.random.random()

# when the accumulated fitness is higher than the random value that index is selected for breeding
while cnt < r:
	# count upward by the accumulating fitness
	cnt += fitnesses[index]
	index += 1

	# avoid floating point issues
	if index == len(fitnesses):
		break

# go back an index
index -= 1

# return agent at index
return agent[index]
```

# Example
Note that this example can be seen in evolution.py.
```python
# callback function to reference the structure of the neural network
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
    # parameters for genetic algorithm and trading bot
    pop_size = 10
    mutation_rate = 0.1
    starting_cash = 1
    trading_fee = 0

    # list of previous prices
    prices = [1, 2, 3, 4, 5, 6]

    # list of inputs for neural network corresponding to prices
    inputs = [[2, 6], [9, 1], [1, 7], [7, 3], [4, 9], [8, 2]]

    # create a population of neural networks to undergo neuro evolution
    pop = Population.Population(pop_size, build_model, mutation_rate, starting_cash, prices[0], trading_fee)

    # constantly evolve the network throughout the programs life
    while True:
        pop.evolve(inputs, prices)
```

# Files
| File | Purpose |
| ------ | ------ |
| evolution | Example of how to run this program.  |
| Population | Creates a population of Agents and contains the methods for genetic algorithms. |
| Agent | Creates a trading agent that makes trading decisions based on its neural network. |
| Wallet | Tracks the money that a trading agent earns throughout its generation. |
