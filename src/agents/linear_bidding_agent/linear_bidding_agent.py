from src.agents.uitls import *
from src.logger import bidding_logger
import os

ROOT_DIR = get_root_dir()
CONFIG_FILE = os.path.join(ROOT_DIR, 'src/agents/config.yaml')


class LinearBiddingAgent:
    def __init__(self, config_file=CONFIG_FILE):
        # hyper parameter
        self.total_budget = get_total_budget(config_file)
        self.target_value = get_target_value(config_file)
        self.T = get_T(config_file)  # total opportunities we have to tune policy

        self.step_t = 1
        self.remaining_budget_t = self.total_budget
        self.remaining_budget_t_minus_1 = self.total_budget
        self.running_budget = self.total_budget
        self.lambda_t = 1.0  # initial action
        self.wr_t = 0.  # the winning rate so far in this episode
        self.r_t = 0.  # total reward so far, in our case, total pctr
        self.cost_t = 0.  # cost within each step
        self.wins_t = 0   # wins within each step
        self.bids_t = 0  # number of bids within each step

        self.total_reward = 0.0
        self.total_wins = 0
        self.total_bids = 0

        self.episode_reward = 0.0
        self.episode_wins = 0
        self.episode_bids = 0

        self.start_date = get_start_date(config_file)  # the date
        self.prev_timestamp = self.start_date

        # Logger
        self.logger = bidding_logger.get_bidding_logger('linear_bidding', file_name='linear_bidding.log')

    def _init_hyper_paras(self, config_file=CONFIG_FILE):
        self.step_t = 1
        self.total_budget = get_total_budget(config_file)
        self.remaining_budget_t = self.total_budget
        self.target_value = get_target_value(config_file)
        self.T = get_T(config_file)  # total opportunities we have to tune policy
        self.lambda_t = 1.0  # initial action
        self.running_budget = self.total_budget

    def _update_reward_cost_within_step(self, reward, cost):
        """
        Update reward and cost within each step
        :param reward:
        :param cost:
        :return:
        """
        self.running_budget -= cost
        self.r_t += reward
        self.cost_t += cost
        self.bids_t += 1
        self.total_bids += 1
        self.total_reward += reward
        self.episode_bids += 1
        self.episode_reward += reward
        if cost > 0.:
            self.wins_t += 1
            self.total_wins += 1
            self.episode_wins += 1

    def _update_step(self):
        """
        Update after each step
        :return:
        """
        self.step_t += 1
        self.remaining_budget_t_minus_1 = self.remaining_budget_t
        self.remaining_budget_t -= self.cost_t
        self.wr_t = self.wins_t * 1.0 / self.bids_t

    def _reset_step(self):
        """
        Reset step
        :return:
        """
        self.r_t = 0.
        self.cost_t = 0.
        self.wins_t = 0
        self.bids_t = 0

    def _reset_episode(self):
        """
        Reset state after each episode
        :return:
        """
        self._init_hyper_paras()
        self.start_date = increment_ts_by_one_day(self.start_date)
        self.prev_timestamp = self.start_date
        self.remaining_budget_t = self.total_budget  # remaining budget
        self.remaining_budget_t_minus_1 = self.total_budget
        self.wr_t = 0.  # the winning rate so far in this episode
        self.r_t = 0.  # total reward so far, in our case, total pctr

        self.cost_t = 0.
        self.wins_t = 0
        self.bids_t = 0

        self.episode_reward = 0.0
        self.episode_wins = 0
        self.episode_bids = 0

    def act(self, obs, reward, cost):
        """

        :param obs:
        :param reward:
        :param cost:
        :return:
        """
        prev_timestamp = self.prev_timestamp
        cur_timestamp = get_timestamp_from_obs(obs)

        interval = int(1440/self.T)
        self.logger.info('pre_time: ' + str(prev_timestamp))
        self.logger.info('cur_time: ' + str(cur_timestamp))
        sd = step_diff(prev_timestamp, cur_timestamp, interval)
        self.logger.info('step_diff: ' + str(sd))
        same_episode = same_date(prev_timestamp, cur_timestamp)
        self.logger.info('same_episode: ' + str(same_episode))
        terminal_state = (same_episode is False)
        self.logger.info('terminal state: ' + str(terminal_state))

        # update reward and cost
        self._update_reward_cost_within_step(reward, cost)

        if sd != 0 and same_episode:
            # update step to prepare for current state
            self._update_step()

            # Update timestamp
            self.prev_timestamp = cur_timestamp
            self._reset_step()

        elif not same_episode:  # episode changes
            self.logger.info('Total reward: ' + str(self.total_reward))
            self.logger.info('Winning rate: ' + str(self.total_wins*1.0/self.total_bids))
            self.logger.info('total bids: ' + str(self.total_bids))

            self.logger.info('Episode reward: ' + str(self.episode_reward))
            self.logger.info('Episode wins: ' + str(self.episode_wins))
            self.logger.info('Episode winning rate: ' + str(self.episode_wins*1./self.episode_bids))
            self.logger.info('Episode bids: ' + str(self.episode_bids))

            self._reset_episode()

        bidding_price = min(self.target_value/self.lambda_t, self.running_budget)
        self.logger.info('running_budget: ' + str(self.running_budget))
        self.logger.info('lambda_t: ' + str(self.lambda_t))
        self.logger.info('bidding price: ' + str(bidding_price))
        self.logger.info('reward/bids ratio: ' + str(self.total_reward*1./self.total_bids))
        self.logger.info('*'*200)
        self.logger.info('*'*200)
        return bidding_price


if __name__ == '__main__':
    print(CONFIG_FILE)
