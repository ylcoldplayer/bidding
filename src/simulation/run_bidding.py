import argparse
import os

from src.environments.bidding_env import BiddingEnv
from src.agents.drl_bidding_agent.drl_bidding_agent import DRLBiddingAgent
from src.agents.linear_bidding_agent.linear_bidding_agent import LinearBiddingAgent
from src.agents.dogd_bidding_agent.dogd_bidding_agent import DOGDBiddingAgent

from src.agents.uitls import get_root_dir

ROOT_DIR = get_root_dir()


def run(agent_type):
    fields = ['campaignId', 'timestamp', 'pctr', 'payprice']
    data_path = 'src/data/test_bidding_request.csv'
    data_path = os.path.join(ROOT_DIR, data_path)
    env = BiddingEnv(data_path, fields)
    agent = DRLBiddingAgent()
    if agent_type == 'drl':
        agent = DRLBiddingAgent()
    elif agent_type == 'linear':
        agent = LinearBiddingAgent()
    elif agent_type == 'dogd':
        agent = DOGDBiddingAgent()
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', help='choose the delegate agent from drl, linear, dogd')
    args = parser.parse_args()
    agent = args.agent
    run(agent)
