from src.logger import bidding_logger


class DOGDBiddingAgent:
    def __init__(self):
        self.logger = bidding_logger.get_bidding_logger('dogd_bidding', file_name='dogd_bidding.log')

    def act(self, obs, reward, cost):
        """

        :param obs:
        :param reward:
        :param cost:
        :return: bidding price
        """
        pass
