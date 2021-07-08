from src.environments.drl_bidding_env import DRLBiddingEnv
from src.agents.drl_bidding_agent.drl_bidding_agent import DRLBiddingAgent


def run():
    fields = ['campaignId', 'timestamp', 'pctr', 'payprice']
    data_path = '../data/test_bidding_request.csv'
    env = DRLBiddingEnv(data_path, fields)
    agent = DRLBiddingAgent()
    obs = env.reset()
    done = False
    reward = 0.0
    cost = 0.0

    while not done:
        action = agent.act(obs, reward, cost)
        next_obs, reward, done, info = env.step(action)
        cost = info['cost']
        obs = next_obs


if __name__ == '__main__':
    run()
