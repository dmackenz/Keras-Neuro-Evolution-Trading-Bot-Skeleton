class Wallet:
    def __init__(self, starting_cash, starting_price, trading_fee):
        self.starting_cash = starting_cash
        self.trading_fee = trading_fee
        self.cash_wallet = starting_cash
        self.btc_wallet = 0
        self.isHolding = False
        self.starting_price = starting_price

    def buy(self, price):
        if not self.isHolding:
            self.btc_wallet = self.cash_wallet / price * (1 - self.trading_fee)
            self.cash_wallet = 0
            self.isHolding = True

    def sell(self, price):
        if self.isHolding:
            self.cash_wallet = self.btc_wallet * price * (1 - self.trading_fee)
            self.btc_wallet = 0
            self.isHolding = False

    def get_holding_earnings(self, final_price):
        return (final_price / self.starting_price) * 100 - 100

    def get_swing_earnings(self, final_price):
        self.sell(final_price)
        return (self.cash_wallet / self.starting_cash) * 100 - 100
