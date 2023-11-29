class StockRawData:
    def __init__(self,date, open, close, max, min,foreign_Buy,foreign_Sell, dealer_Buy, dealer_Sell, investment_Buy, investment_Sell):
        self.date = date
        self.open = open
        self.close = close
        self.max = max
        self.min = min
        self.foreign_Buy = foreign_Buy
        self.foreign_Sell = foreign_Sell
        self.dealer_Buy = dealer_Buy
        self.dealer_Sell = dealer_Sell
        self.investment_Buy = investment_Buy
        self.investment_Sell = investment_Sell
     