class Wallet:
    def __init__(self, starting_cash, starting_price, trading_fee):
        self.starting_cash = float(starting_cash)
        self.trading_fee = float(trading_fee)
        self.cash_wallet = float(starting_cash)
        self.btc_wallet = float(0)
        self.isHolding = False
        self.starting_price = starting_price

    def buy(self, price):
        if not self.holding():
            price = float(price)

            self.btc_wallet = self.cash_wallet / price * (1 - self.trading_fee)

            self.cash_wallet = 0
            self.isHolding = True

    def sell(self, price):
        if self.holding():
            price = float(price)
            
            new_cash_wallet = self.btc_wallet * price * (1 - self.trading_fee)
            self.cash_wallet = new_cash_wallet

            self.btc_wallet = 0
            self.isHolding = False

    def get_cash_wallet(self):
        return self.cash_wallet

    def get_btc_wallet(self):
        return self.btc_wallet

    def get_holding_earnings(self, final_price):
        return (float(final_price) / self.starting_price) * 100 - 100

    def get_swing_earnings(self, final_price):
        self.sell(final_price)
        return (self.cash_wallet / self.starting_cash) * 100 - 100

    def holding(self):
        return self.isHolding

if __name__ == '__main__':
    starting_cash = 10
    starting_price = 10
    trading_fee = 0.01

    wallet = Wallet(starting_cash, starting_price, trading_fee)

    wallet.buy(15)
    wallet.sell(20)

    print(wallet.get_swing_earnings(-50))