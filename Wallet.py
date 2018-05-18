class Wallet:
    def __init__(self, starting_cash, starting_price, trading_fee):
        self.starting_cash = starting_cash
        self.trading_fee = trading_fee
        self.cash_wallet = starting_cash
        self.btc_wallet = 0
        self.isHolding = False
        self.starting_price = starting_price
        self.current_buy = None
        self.old_cash_wallet = 0
        self.cash_history = []
        self.trade_history = []

    def buy(self, idx, price):
        if not self.isHolding:
            self.current_buy = price
            
            self.btc_wallet = self.cash_wallet / price * (1 - self.trading_fee)
            
            self.old_cash_wallet = self.cash_wallet
            self.cash_wallet = 0
            self.isHolding = True

    def sell(self, idx, price):
        if self.isHolding:
            self.cash_wallet = self.btc_wallet * price * (1 - self.trading_fee)

            self.cash_history.append([idx, self.cash_wallet])
            self.trade_history.append(self.cash_wallet / self.old_cash_wallet * 100 - 100)


            self.btc_wallet = 0
            self.isHolding = False

    def get_holding_earnings(self, final_price):
        return (final_price / self.starting_price) * 100 - 100

    def get_swing_earnings(self, idx, final_price):
        self.sell(idx, final_price)
        return (self.cash_wallet / self.starting_cash) * 100 - 100

if __name__ == '__main__':
    starting_cash = 10
    starting_price = 10
    trading_fee = 0.01

    wallet = Wallet(starting_cash, starting_price, trading_fee)

    wallet.buy(15)
    wallet.sell(20)

    print(wallet.get_swing_earnings(-50))
