import Agent
import numpy as np

class Population(object):
    def __init__(self, pop_size, model_builder, mutation_rate, starting_cash, starting_price, trading_fee):
        self.pop_size = pop_size
        self.agents = []

        self.model_builder = model_builder
        self.mutation_rate = mutation_rate
        self.starting_cash = starting_cash
        self.starting_price = starting_price
        self.trading_fee = trading_fee

        for i in range(self.pop_size):
            agent = Agent.Agent(self.model_builder, self.mutation_rate, self.starting_cash, self.starting_price, self.trading_fee, i)
            self.agents.append(agent)

    def evolve(self, inputs_list, prices_list):
        self.batch_feed_inputs(inputs_list, prices_list)
        self.normalize_fitness()

        output_str = "profits: "
        for agent in self.agents:
            output_str += "{0:.2f}%".format(agent.get_score()).ljust(10)

        print(output_str)

        self.generate_next_generation()

    def batch_feed_inputs(self, inputs_list, prices_list):
        for i in range(len(self.agents)):
            self.agents[i].batch_act(inputs_list, prices_list)

    def normalize_fitness(self):
        s = 0
        for i in range(len(self.agents)):
            s += self.agents[i].get_score()

        for i in range(len(self.agents)):
            if s != 0:
                fit = self.agents[i].get_score() / s
                self.agents[i].set_fitness(fit)
            else:
                self.agents[i].set_fitness(0)

    def pool_selection(self):
        index = 0
        r = np.random.random()

        while r > 0:
            r -= self.agents[index].get_fitness()
            index += 1

            if index == len(self.agents):
                break

        index -= 1

        return self.agents[index].model

    def generate_next_generation(self):
        for i in range(self.pop_size):
            model = self.pool_selection()
            self.agents[i] = Agent.Agent(self.model_builder, self.mutation_rate, self.starting_cash, self.starting_price, self.trading_fee, i, inherited_model=model)
