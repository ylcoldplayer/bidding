from src.agents.drl_bidding_agent.dqn import DQNAgent
from src.agents.drl_bidding_agent.reward_net import RewardNetAgent
from src.agents.drl_bidding_agent.uitls import *
import numpy as np

# TODO: assign
CONFIG_FILE = ''


class DRLBiddingAgent:
    def __init__(self, ):
        self.betas = [-0.08, -0.03, -0.01, 0, 0.01, 0.03, 0.08]
        self.eps_high = 0.95
        self.eps_low = 0.05
        self.anneal = 0.00005
        self.prev_timestamp = 0  # todo:
        self._reset_episode()

        self.dqn_agent = DQNAgent(state_size=self.dqn_state_size, action_size=self.dqn_action_size)
        self.reward_net_agent = RewardNetAgent(state_action_size=8)  # Todo: implement
        self.date = 0  # the date

        self.dqn_prev_state = None
        self.dqn_prev_action = 3

        self.total_reward = 0.0
        self.totoal_wins = 0
        self.total_bids = 0

        self.T = 96  # number of time steps

    # Todo: implement
    def _init_hyper_paras(self, config_file=CONFIG_FILE):
        self.step_t = 1
        self.total_budget = None
        self.remaining_budget_t = self.total_budget
        self.target_value = None
        self.gamma = 1  # discount factor
        self.T = None  # total opportunities we have to tune policy
        self.dqn_state_size = None
        self.dqn_action_size = None
        self.dqn_prev_state = None
        self.dqn_prev_action = 3
        self.lambda_t = 1.0  # initial action
        self.running_budget = self.total_budget

    def _get_state(self):
        """
        Get current state
        :return:
        """
        state_array = [
            self.step_t,
            self.remaining_budget_t,
            self.rol_t,
            self.bct_t,
            self.cpm_t,
            self.wr_t,
            self.r_t
        ]
        return np.asarray(state_array)

    def _reset_episode(self):
        """
        Reset state after each episode
        :return:
        """
        self._init_hyper_paras()
        self.prev_timestamp = 0
        self.remaining_budget_t = self.total_budget  # remaining budget
        self.remaining_budget_t_minus_1 = self.total_budget
        self.bct_t = 0.  # budget consumption rate
        self.rol_t = self.T  # regulation opportunities left
        self.cpm_t = 0.  # the cost per million of impression of the winning impressions
        self.wr_t = 0.  # the winning rate so far in this episode
        self.r_t = 0.  # total reward so far, in our case, total pctr

        self.cost_t = 0.
        self.wins_t = 0
        self.bids_t = 0

        self.eps = self.eps_high

    def _update_reward_cost_within_step(self, reward, cost):
        """
        Update reward and cost within each step
        :param reward:
        :param cost:
        :return:
        """
        self.running_budget -= cost
        self.r_t += reward
        self.total_reward += reward
        self.cost_t += cost
        self.bids_t += 1
        self.total_bids += 1
        if cost > 0.:
            self.wins_t += 1
            self.totoal_wins += 1

    def _reset_step(self):
        """
        Reset step
        :return:
        """
        self.r_t = 0.
        self.cost_t = 0.
        self.wins_t = 0
        self.bids_t = 0

        self.eps = max(self.eps - self.anneal * self.step_t, self.eps_low)

    def _update_step(self):
        """
        Update after each step
        :return:
        """
        self.step_t += 1
        self.remaining_budget_t_minus_1 = self.remaining_budget_t
        self.remaining_budget_t -= self.cost_t
        self.bct_t = (self.remaining_budget_t_minus_1 - self.remaining_budget_t) * 1.0 / self.remaining_budget_t_minus_1
        self.rol_t -= 1
        self.cpm_t = self.cost_t * 1.0 / self.bids_t
        self.wr_t = self.wins_t * 1.0 / self.bids_t

    def act(self, obs, reward, cost):
        """
        Agent performs learning based on reward and cost from last action AND chooses the next action
        :param obs: current observation
        :param reward: reward from last action
        :param cost: cost from last action
        :return:
        """
        """
        Paper: https://arxiv.org/pdf/1802.08365.pdf
        
        Algorithm: 
            1. Get previous state S_prev from agent, get previous timestamp
            2. Get current timestamp, t_current (from current obs)
                update reward, cost, winning, etc  
                    
                if t_current != S_prev AND still in current episode
                    1. get previous action, A_pre 
                    2. update RewardNet in this step
                        a. learn net, sgd
                        b. store pair (S_prev, A_prev) in S
                    3. update and get current state S_cur
                        a. t_current = S_prev + 1
                        b. t_current > S_prev + 1 
                    4. get previous reward, R_pre from RewardNet
                    5. store tuple (S_prev, A_prev, R_prev, S_cur) in replay
                    6. train DQNAgent
                    
                    
                if t_episode changes:
                    1. Update RewardNet target M
                    2. Reset 
        """
        prev_timestamp = self.prev_timestamp
        cur_timestamp = get_timestamp_from_obs(obs)
        dqn_pre_state = self.dqn_prev_state

        interval = int(1440/self.T)
        sd = step_diff(prev_timestamp, cur_timestamp, interval)
        same_episode = same_date(prev_timestamp, cur_timestamp)
        terminal_state = (same_episode is False)

        # update reward and cost
        self._update_reward_cost_within_step(reward, cost)

        # TODO: Should consider the scenario in which two bidding requests more than 15 minutes away from each other
        if sd != 0 and same_episode:
            dqn_prev_action = self.dqn_prev_action

            # Update RewardNet
            self.reward_net_agent.perform_mini_batch()
            self.reward_net_agent.add_pair_t(state=dqn_pre_state, action=dqn_prev_action)
            self.reward_net_agent.add_reward_t(self.r_t)

            # update step to prepare for current state
            self._update_step()
            # Observe state s_t
            dqn_cur_sate = self._get_state()

            # Choose action for current state using adaptive epsilon-greedy policy, i.e. get scale factor beta
            beta_t_idx = self.dqn_agent.act(dqn_cur_sate, eps=self.eps)
            # Update lambda_t
            self.lambda_t *= (1 + self.betas[beta_t_idx])

            # Get RewardNet reward_net_r_t
            sa = np.append(dqn_pre_state, dqn_prev_action)
            reward_net_r_t = float(self.reward_net_agent.get_reward_net_r_t(sa))

            # sample mini batch apply gradient descent to update Q function, store experience in replay buffer
            self.dqn_agent.update(dqn_cur_sate, dqn_prev_action, reward_net_r_t, dqn_cur_sate, terminal_state)

            # Update dqn state and action
            self.dqn_prev_state = dqn_cur_sate
            self.dqn_prev_action = beta_t_idx

            # Update timestamp
            self.prev_timestamp = cur_timestamp
            self._reset_step()

        elif not same_episode:  # episode changes
            self.reward_net_agent.update_episode()
            self.reward_net_agent.reset_episode()

        bidding_price = min(self.target_value/self.lambda_t, self.running_budget)
        return bidding_price


if __name__ == '__main__':
    # TODO: add test code here
    pass
