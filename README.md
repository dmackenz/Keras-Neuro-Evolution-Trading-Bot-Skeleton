# Keras-Neuro-Evolution-Trading-Bot-Skeleton   

**Credit goes to [dmackenz](https://github.com/dmackenz)**   

This project outlines the skeleton for creating a neuro-evolution trading bot with a Keras neural network. This is part of a reinforcement learning strategy to "reward" the neural network whenever it creates a trading strategy that generates profit. After each generation of neural networks, the highest profit networks will carry on to the next generation and produce a new generation of networks that are slightly mutated based on those with the highest profits. This mutation will create new trading strategies based on the inputs provided. If the mutation is not aligned with the end goal of generating profit, then it most likely will not carry on to the next generation.

This is in early stages of development and has only proven that it is capable of evolving a neural network to generate profit in a trading bot scenario. This project does not directly connect to an exchange API (but could be easily extendable via the Wallet class). I am not responsible for other's use cases of this program and the money that they make or lose with this program.

## Fitness Normalization Algorithm (pseudo-code)
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

## Selection Algorithm (pseudo-code)
```Python
idx, cnt = 0, 0

# sort fitnesses in descending order
fitnesses.sort(reversed=True)

# generate random number between 0 and 1
r = np.random.random()

# when the accumulated fitness is higher than the random value that index is selected for breeding
while cnt < r and idx != len(self.agents):
    cnt += self.agents[idx].fitness
    idx += 1

# return index of agent
return idx - 1
```

## Example
Note that this example can be seen in evolution.py.
```python
# callback function to reference the structure of the neural network
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
    starting_cash = 1.
    trading_fee = 0
    generations = 3

    # generate random test data
    test_size = 100
    np.random.seed(42)
    prices = np.random.randint(20, size=test_size) + 1
    inputs = np.random.rand(test_size, 4) * 2 - 1

    # build initial population
    pop = Population(pop_size, build_model, mutation_rate, 
                     mutation_scale, starting_cash, prices[0], trading_fee)

    # run defined number of evolutions
    for i in range(generations):
        start = time()
        pop.evolve(inputs, prices)
        print('\n\nDuration: {0:.2f}s'.format(time()-start))
```

## Files
| File | Purpose |
| ------ | ------ |
| evolution | Example of how to run this program.  |
| Population | Creates a population of Agents and contains the methods for genetic algorithms. |
| Agent | Creates a trading agent that makes trading decisions based on its neural network. |
| Wallet | Tracks the money that a trading agent earns throughout its generation. |
