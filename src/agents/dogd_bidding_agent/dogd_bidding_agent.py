from src.logger import bidding_logger


class DOGDBiddingAgent:
    def __init__(self):
        self.logger = bidding_logger.get_bidding_logger('dogd_bidding', file_name='dogd_bidding.log')
