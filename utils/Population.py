from utils.Agent import Agent
from keras import backend as K
from keras.models import load_model, model_from_json

import matplotlib.pyplot as plt

import numpy as np
import os

class Population(object):
    def __init__(self, pop_size, model_builder, mutation_rate, mutation_scale, starting_cash, starting_price, trading_fee, big_bang=True):
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

        if big_bang == True:
            for i in range(self.pop_size):
                print("\rbuilding agents {:.2f}%...".format((i + 1) / self.pop_size * 100), end="")
                agent = Agent(self, i)
                self.agents.append(agent)

    def set_preexisting_agent_base(self, model):
        self.agents = []
        for i in range(self.pop_size):
            self.agents.append(Agent.Agent(self, i, inherited_model=model))

    def evolve(self, inputs_list, prices_list, output_width=5, plot_best=False):
        print("\n\n======================\ngeneration number {}\n======================".format(self.generation_number))

        self.batch_feed_inputs(inputs_list, prices_list)
        self.print_profits(output_width, prices_list)
        self.normalize_fitness()
        self.sort_by_decreasing_fitness()
        if plot_best == True:
            self.plot_best_agent(prices_list)
        self.save_best_agent()
        self.generate_next_generation()

    def batch_feed_inputs(self, inputs_list, prices_list):
        for i in range(len(self.agents)):
            self.agents[i].batch_act(inputs_list, prices_list)
            print("\rFeeding inputs {:.2f}%...".format((i + 1) / self.pop_size * 100), end="")

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

        for i in range(self.pop_size):
            if s != 0:
                self.agents[i].fitness = self.agents[i].score / s
            else:
                self.agents[i].fitness = 0

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

    def print_profits(self, output_width, prices):
        c = 0
        profit_arr = []
        for agent in self.agents:
            profit_arr.append(agent.wallet.get_swing_earnings(len(prices), prices[-1]))
        profit_arr.sort()

        output_str = "\naverage profit: {0:.2f}%\n".format(np.average(profit_arr))
        for score in profit_arr:
            output_str += "{0:.2f}%".format(score).ljust(20)
            c += 1
            if c % output_width == 0:
                output_str += "\n"
        print(output_str)

    def save_best_agent(self):
        self.sort_by_decreasing_fitness()
        self.agents[0].save("saved_agent/best_agent")

    def plot_best_agent(self, prices):
        indexes, wallet_values = [], []
        for hist in self.agents[0].wallet.cash_history:
            indexes.append(hist[0])
            wallet_values.append(hist[1])

        plt.figure(1)
        plt.suptitle("Trading Bot Generation {}".format(self.generation_number))

        ax1 = plt.subplot(211)
        ax1.set_ylabel("Price Graph")
        ax1.plot(prices)

        ax2 = plt.subplot(212, sharex=ax1)
        ax2.set_ylabel("Cash Wallet Value")
        ax2.plot(indexes, wallet_values)

        plt.show()