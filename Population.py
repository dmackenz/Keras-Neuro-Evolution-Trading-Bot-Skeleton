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
        self.generation_number = 1
        self.output_width = 5

        for i in range(self.pop_size):
            print("\rbuilding agents {:.2f}%...".format(i + 1 / self.pop_size * 100), end="")
            agent = Agent.Agent(self.model_builder, self.mutation_rate, self.starting_cash, self.starting_price, self.trading_fee, i)
            self.agents.append(agent)
        print("done")


    def evolve(self, inputs_list, prices_list):
        c = 0
        scores = []
        
        print("\n\n=================\ngeneration number {}\n=================".format(self.generation_number))
        print("feeding inputs...")
        self.batch_feed_inputs(inputs_list, prices_list)
        
        print("normalizing fitness...")
        self.normalize_fitness()

        for agent in self.agents:
            scores.append(agent.get_score())
        scores.sort()
        output_str = "\nprofits:\n"
        for score in scores:
            output_str += "{0:.2f}%".format(score).ljust(10)
            c += 1
            if c % self.output_width == 0:
                output_str += "\n"
        print(output_str)
        print("average score: {0:.2f}%".format(float(sum(scores)) / float(len(scores))))

        self.generate_next_generation()

    def batch_feed_inputs(self, inputs_list, prices_list):
        for i in range(len(self.agents)):
            self.agents[i].batch_act(inputs_list, prices_list)

    def normalize_fitness(self):
        s = 0
        for i in range(len(self.agents)):
            if self.agents[i].get_score() > 0:
                s += self.agents[i].get_score()

        for i in range(len(self.agents)):
            if s != 0 and self.agents[i].get_score() > 0:
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
            print("\rcreating next generation {0:.2f}%...".format(i / self.pop_size * 100), end="")
            model = self.pool_selection()
            self.agents[i] = Agent.Agent(self.model_builder, self.mutation_rate, self.starting_cash, self.starting_price, self.trading_fee, i, inherited_model=model)
        self.generation_number += 1
        print("done")
