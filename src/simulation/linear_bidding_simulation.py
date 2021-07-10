from src.environments.bidding_env import BiddingEnv
from src.agents.linear_bidding_agent.linear_bidding_agent import LinearBiddingAgent


def run():
    fields = ['campaignId', 'timestamp', 'pctr', 'payprice']
    data_path = '../data/test_bidding_request.csv'
    env = BiddingEnv(data_path, fields)
    agent = LinearBiddingAgent()
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
