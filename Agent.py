from utils.Wallet import Wallet
from keras.models import clone_model
import numpy as np

class Agent(object):
    def __init__(self, population, agent_id, inherited_model=None):
        self.population = population
        self.wallet = Wallet(self.population.starting_cash, self.population.starting_price, self.population.trading_fee)

        self.agent_id = agent_id

        self.score = 0
        self.fitness = 0
        self.model = None

        self.BUY = 1
        self.SELL = 2
        self.SLEEP = 3

        # if mutating from an existing model
        if inherited_model:
            self.model = inherited_model
            self.mutate()
        else:
            self.model = self.population.model_builder()

    def batch_encode_prediction(self, predictions):
        # converting output to trade signals
        encodeds = []

        if self.population.has_one_output == True:
            for prediction in predictions:
                if prediction[0] >= 0:
                    encodeds.append(self.BUY)
                else:
                    encodeds.append(self.SELL)
        else:
            if (self.population.has_sleep_functionality):
                for prediction in predictions:
                    if np.argmax(prediction) == 0: encodeds.append(self.BUY)
                    elif np.argmax(prediction) == 1: encodeds.append(self.SELL)
                    else: encodeds.append(self.SLEEP)
            else:
                for prediction in predictions:
                    if np.argmax(prediction) == 0: encodeds.append(self.BUY)
                    else: encodeds.append(self.SELL)

        return encodeds

    def batch_act(self, inputs, prices):
        predictions = self.model.predict(np.array(inputs))
        encodeds = self.batch_encode_prediction(predictions)

        # simulate trades based on trade signals
        for idx, encoded in enumerate(encodeds):
            if encoded == self.BUY:
                self.wallet.buy(idx, prices[idx])
            elif encoded == self.SELL:
                self.wallet.sell(idx, prices[idx])

        # evaluate score
        self.score = self.wallet.get_swing_earnings(idx, prices[-1])

    def save(self, filename):
        model_json = self.model.to_json()
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename + ".json", "w") as json_file:
            json_file.write(model_json)
        self.model.save_weights(filename + ".h5")

    def mutate(self):
        # iterate through each layer of model
        for i in range(len(self.model.layers)):
            weights = self.model.layers[i].get_weights()

            # mutate weights of network
            for j in range(len(weights[0])):
                for k in range(len(weights[0][j])):

                    # randomly mutate based on mutation rate
                    if np.random.random() < self.population.mutation_rate:
                        weights[0][j][k] += np.random.normal(scale=self.population.mutation_scale) * 0.5

            self.model.layers[i].set_weights(weights)
