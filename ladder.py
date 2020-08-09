class Ladder:
    def __init__(self, slots):

        self.queue = ["SHY"] * slots

    def update(self, ticker):

        queue = self.queue
        queue.pop(0)
        queue = queue.append(ticker)

    def get_queue(self):
        return self.queue

#example
# =============================================================================
# ladder = Ladder(4)
# ladder.update("AAPL")
# ladder.update("SQ")
# print(ladder.get_queue())
# =============================================================================