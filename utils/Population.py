from utils.Agent import Agent
import numpy as np


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
            agent = Agent(self, i)
            self.agents.append(agent)

    def evolve(self, inputs_list, prices_list):
        self.batch_feed_inputs(inputs_list, prices_list)
        self.print_scores()
        self.normalize_fitness()
        self.sort_by_decreasing_fitness()
        self.generate_next_generation()

    def batch_feed_inputs(self, inputs_list, prices_list):
        print("\n\n===================\n"+
                  "generation number {}\n".format(self.generation_number)+
                  "===================\n"+
                  "feeding inputs...")
        
        for i in range(len(self.agents)):
            self.agents[i].batch_act(inputs_list, prices_list)

    def normalize_fitness(self):
        print("normalizing fitness...")

        scores_arr = []
        for agent in self.agents:
            scores_arr.append(agent.score)
        
        mi = min(scores_arr)
        ma = max(scores_arr)
        den = ma - mi

        for i in range(len(self.agents)):
            new_score = ((self.agents[i].score - mi) / den) ** 2
            self.agents[i].score = new_score

        s = 0
        for agent in self.agents:
            s += agent.score

        for i in range(1, len(self.agents)):
            fit = self.agents[i].score / s
            self.agents[i].fitness = fit

    def pool_selection(self):
        index = 0
        r = np.random.random()
        cnt = 0

        while cnt < r:
            cnt += self.agents[index].fitness
            index += 1

            if index == len(self.agents):
                break

        index -= 1

        return self.agents[index].model

    def generate_next_generation(self):
        newAgents = []
        for i in range(len(self.agents)):
            print("\rcreating next generation {0:.2f}%...".format((i + 1) / self.pop_size * 100), end="")
            model = self.pool_selection()
            newAgents.append(Agent(self, i, inherited_model=model))

        self.agents = newAgents
        self.generation_number += 1

    def sort_by_decreasing_fitness(self):
        self.agents.sort(key=lambda x: x.fitness, reverse=True)        

    def print_scores(self):
        c = 0
        scores_arr = []

        for agent in self.agents:
            scores_arr.append(agent.score)
        
        scores_arr.sort()

        output_str = "\naverage score: {0:.2f}%\n".format(np.average(scores_arr))
        for score in scores_arr:
            output_str += "{0:.2f}%".format(score).ljust(10)
            c += 1
            if c % self.output_width == 0:
                output_str += "\n"
        print(output_str)

    def print_fitnesses(self):
        s = 0
        for idx, agent in enumerate(self.agents):
            print(idx, agent.agent_id, agent.fitness)
            s += agent.fitness
