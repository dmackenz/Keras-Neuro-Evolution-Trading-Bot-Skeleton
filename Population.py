import Agent
import numpy as np
import copy

class Population(object):
    def __init__(self, pop_size, model_builder, mutation_rate, mutation_scale, starting_cash, starting_price, trading_fee):
        self.pop_size = pop_size
        self.agents = []

        self.model_builder = model_builder
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.starting_cash = starting_cash
        self.starting_price = starting_price
        self.trading_fee = trading_fee

        self.generation_number = 1
        self.output_width = 5

        for i in range(self.pop_size):
            print("\rbuilding agents {:.2f}%...".format((i + 1) / self.pop_size * 100), end="")
            agent = Agent.Agent(self, i)
            self.agents.append(agent)

    def evolve(self, inputs_list, prices_list):
        self.batch_feed_inputs(inputs_list, prices_list)
        self.print_scores()
        self.normalize_fitness()
        self.sort_by_decreasing_fitness()
        self.generate_next_generation()

    def batch_feed_inputs(self, inputs_list, prices_list):
        print("\n\n=================\ngeneration number {}\n=================".format(self.generation_number))
        print("feeding inputs...")
        for i in range(len(self.agents)):
            self.agents[i].batch_act(inputs_list, prices_list)

    def normalize_fitness(self):
        print("normalizing fitness...")

        scores_arr = []
        for agent in self.agents:
            scores_arr.append(copy.deepcopy(agent.get_score()))
        mi = min(scores_arr)
        ma = max(scores_arr)
        den = float(ma - mi)

        for i in range(len(self.agents)):
            new_score = (float(self.agents[i].get_score() - mi) / den) ** 2
            self.agents[i].set_score(new_score)

        s = 0
        for agent in self.agents:
            s += agent.get_score()

        s = float(s)
        for i in range(len(self.agents)):
            if s != 0:
                fit = self.agents[i].get_score() / s
                self.agents[i].set_fitness(fit)

    def pool_selection(self):
        index = 0
        r = np.random.random()
        cnt = 0

        while cnt < r:
            cnt += self.agents[index].get_fitness()
            index += 1

            if index == len(self.agents):
                break

        index -= 1

        return self.agents[index].model

    def generate_next_generation(self):
        newAgents = []
        for i in range(len(self.agents)):
            print("\rcreating next generation {0:.2f}%...".format(i / self.pop_size * 100), end="")
            model = self.pool_selection()
            newAgents.append(Agent.Agent(self, i, inherited_model=model))
        self.agents = newAgents
        self.generation_number += 1


    def sort_by_decreasing_fitness(self):
        self.agents.sort(key=lambda x: x.fitness, reverse=True)        

    def print_scores(self):
        c = 0
        scores_arr = []

        for agent in self.agents:
            scores_arr.append(copy.deepcopy(agent.get_score()))
        scores_arr.sort()

        avg = float(sum(scores_arr)) / float(len(scores_arr))

        output_str = "\naverage score: {0:.2f}%\n".format(avg)
        for score in scores_arr:
            output_str += "{0:.2f}%".format(score).ljust(10)
            c += 1
            if c % self.output_width == 0:
                output_str += "\n"
        print(output_str)

    def print_fitnesses(self):
        s = 0
        for idx, agent in enumerate(self.agents):
            print(idx, agent.agent_id, agent.get_fitness())
            s += agent.get_fitness()
