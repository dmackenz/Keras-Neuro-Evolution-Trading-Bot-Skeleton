import Wallet
import copy
from keras.models import clone_model
import numpy as np

class Agent(object):
    def __init__(self, model_builder, mutation_rate, mutation_scale, starting_cash, starting_price, trading_fee, agent_id, inherited_model=None):
        self.model_builder = model_builder
        self.mutation_rate = mutation_rate
        self.mutation_scale = mutation_scale
        self.wallet = Wallet.Wallet(starting_cash, starting_price, trading_fee)
        self.score = 0
        self.fitness = 0
        self.agent_id = agent_id

        self.BUY = 1
        self.SELL = 2
        self.SLEEP = 3

        if inherited_model:
            model_copy = clone_model(inherited_model)
            model_copy.set_weights(inherited_model.get_weights())
            self.model = model_copy
            self.mutate()
        else:
            self.model = self.model_builder()

    def update_score(self, price):
        self.score = self.wallet.get_swing_earnings(price)

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def get_fitness(self):
        return self.fitness

    def set_fitness(self, fitness):
        self.fitness = fitness

    def batch_encode_prediction(self, predictions):
        encodeds = []

        for prediction in predictions:
            buy_prob = prediction[0]
            sell_prob = prediction[1]
            sleep_prob = prediction[2]

            if (buy_prob > sell_prob) and (buy_prob > sleep_prob):
                encodeds.append(self.BUY)
            elif (sell_prob > buy_prob) and (sell_prob > sleep_prob):
                encodeds.append(self.SELL)
            elif (sleep_prob > sell_prob) and (sleep_prob > buy_prob):
                encodeds.append(self.SLEEP)
            else:
                encodeds.append(self.SLEEP)

        return encodeds

    def batch_act(self, inputs, prices):
        inputs = np.array(inputs)
        predictions = self.model.predict(inputs)
        encodeds = self.batch_encode_prediction(predictions)

        for idx, encoded in enumerate(encodeds):
            if encoded == self.BUY:
                self.wallet.buy(prices[idx])
            elif encoded == self.SELL:
                self.wallet.sell(prices[idx])

        self.update_score(prices[-1])

    def save(self, filename):
        self.model.save_model(filename)

    def mutate(self):
        for i in range(len(self.model.layers)):
            weights = self.model.layers[i].get_weights()

            # mutate weights of network
            # currently only tested on Dense
            for j in range(len(weights[0])):
                for k in range(len(weights[0][j])):
                    if np.random.random() < self.mutation_rate:
                        weights[0][j][k] += np.random.normal(scale=self.mutation_scale)

            self.model.layers[i].set_weights(weights)