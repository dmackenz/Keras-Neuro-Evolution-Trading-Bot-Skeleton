from utils.Agent import Agent
from keras import backend as K
from keras.models import load_model
import numpy as np
from keras.models import model_from_json
import os


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
        idx, cnt = 0, 0
        r = np.random.random()

        while cnt < r and idx != len(self.agents):
            cnt += self.agents[idx].fitness
            idx += 1

        return idx - 1

    def generate_next_generation(self):
        # find models to mutate to next generation
        newAgents_idx = []
        for i in range(self.pop_size):
            newAgents_idx.append(self.pool_selection())

        # temporarily save models and clear session
        configs, weights = [], []    
        for model_idx in newAgents_idx:
            configs.append(self.agents[model_idx].model.to_json())
            weights.append(self.agents[model_idx].model.get_weights())

        K.clear_session()

        # reload models
        newAgents = []
        for i in range(self.pop_size):
            print("\rcreating next generation {0:.2f}%...".format((i + 1) / self.pop_size * 100), end="")
            loaded = model_from_json(configs[i])
            loaded.set_weights(weights[i])
            newAgents.append(Agent(self, i, inherited_model=loaded))

        self.agents = newAgents
        self.generation_number += 1

        # mutation scale decay
        self.mutation_scale *= 0.95

    def sort_by_decreasing_fitness(self):
        self.agents.sort(key=lambda x: x.fitness, reverse=True)        

    def print_scores(self):
        scores_arr = []
        for agent in self.agents:
            scores_arr.append(agent.score)

        print("\naverage score: {0:.2f}%".format(np.average(scores_arr)))
        print("largest score: {0:.2f}%\n".format(max(scores_arr)))

    def print_fitnesses(self):
        s = 0
        for idx, agent in enumerate(self.agents):
            print(idx, agent.agent_id, agent.fitness)
            s += agent.fitness
