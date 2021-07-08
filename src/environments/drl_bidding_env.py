import pandas as pd
import gym


class DRLBiddingEnv(gym.Env):
    def __init__(self, data_path, fields):
        """

        :param data_path:
        :param fields:
        """
        self._bidding_data = BiddingData(data_path, fields)
        self._step = 0
        self._total_bids = self._bidding_data.total_bids()

    def reset(self):
        """
        Reset environment after every episode
        :return: initial observation, in bidding setting, it's the first bidding request
        """
        self._step = 0
        br = self._get_bid_request()
        return self._get_obs(br)

    def step(self, action):
        """
        Accepts an action and returns a tuple (observation, reward, done, info).
        Here action is the bidding price
        :param action:
        :return:
        """
        # get current bidding request
        cur_br = self._get_bid_request()

        # calculate reward, cost after this action
        reward, cost = self._get_reward_cost(cur_br, action)
        info = {'cost': cost}

        # determine if reach end of episode
        done = self._step == (self._total_bids-1)

        # get next obs
        next_br = self._get_next_bid_request()
        next_obs = self._get_obs(next_br)

        # move one step forward
        self._step += 1
        return next_obs, reward, done, info

    @staticmethod
    def _get_obs(br):
        """
        Get observation at current step

        :return: Observation map at this step
        """
        obs = {}
        if br is not None:
            obs['timestamp'] = br['timestamp']
            obs['pctr'] = br['pctr']
            obs['payprice'] = br['payprice']
        return obs

    @staticmethod
    def _get_reward_cost(br, action):
        if br is None:
            raise TypeError('bidding request should not be None type')
        pctr = br['pctr']
        payprice = br['payprice']

        if action > payprice:
            reward = pctr
            cost = payprice
        else:
            reward = 0
            cost = 0
        return reward, cost

    def _get_bid_request(self):
        """
        Get current bidding request
        :return:
        """
        return self._bidding_data.get_request(self._step)

    def _get_next_bid_request(self):
        """
        Get next bidding request if there is any
        :return:
        """
        if self._step+1 <= self._total_bids-1:
            return self._bidding_data.get_request(self._step+1)
        else:
            return None

    def close(self):
        super().close()

    def seed(self, seed=None):
        return super().seed(seed)

    def render(self, mode='human'):
        pass


class BiddingData:
    """
    Each bidding request should contain all information we need for auction
    """
    def __init__(self, data_path, fields):
        """

        :param data_path: data source file path
        :param fields: ['campaignId', 'timestamp', 'pctr', 'payprice']
        """
        self._bidding_requests = self._load_br_from_data(data_path, fields)

    def get_request(self, step):
        return self._bidding_requests.iloc[step, :]

    def _load_br_from_data(self, data_path, fields):
        """
        load bidding requests from local directory
        :param data_path:
        :param fields:
        :return: pandas data frame of bidding request
        """
        bidding_requests = pd.read_csv(data_path)[fields]
        return bidding_requests

    def total_bids(self):
        """
        :return: number of total bidding requests
        """
        return len(self._bidding_requests.index)


if __name__ == '__main__':
    fields = ['campaignId', 'timestamp', 'pctr', 'payprice']
    data_file_pth = '../data/test_bidding_request.csv'
    env = DRLBiddingEnv(data_file_pth, fields)
    print('next bid request: \n', env._get_next_bid_request())
    print('\n')
    print(env.step(4.0))
    print(env.step(1.0))
    print(env.step(5))
