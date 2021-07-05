import random

from src.agents.drl_bidding_agent.uitls import *
from numpy.random import lognormal, normal

START_DATE = '2021-07-04T00::00::45.456777'
EPISODES = 5
STEPS = 96

CAMPAIGN_ID = 'campaignId'
TIMESTAMP = 'timestamp'
PCTR = 'pctr'
PAY_PRICE = 'payprice'

FIELDS = [CAMPAIGN_ID, TIMESTAMP, PCTR, PAY_PRICE]


# TODO: generate in batch manner to get better efficiency
def generate(start_date, episodes, steps, output_file):
    """
    Generate dummy data for training
    :param start_date:
    :param episodes:
    :param steps:
    :param output_file:
    :return:
    """
    header = ','.join(FIELDS) + '\n'
    with open(output_file, 'w') as fd:
        fd.write(header)
        data_arrays = []
        first_dp = generate_data_point(start_date)
        data_arrays.append(first_dp)

        interval = int(1440/steps)
        cur_ts = increment_ts(start_date, interval)

        while step_diff(start_date, cur_ts, interval) < steps*episodes:
            dp_tmp = generate_data_point(cur_ts)
            data_arrays.append(dp_tmp)
            cur_ts = increment_ts(cur_ts, interval)

        for dp in data_arrays:
            row = ','.join(dp) + '\n'
            fd.write(row)


def generate_data_point(ts):
    """
    Generate one data point
    :param ts:
    :return:
    """
    campaign_id = '1'
    pctr = generate_pctr()
    payprice = generate_payprice()
    return [campaign_id, ts, str(pctr), str(payprice)]


def generate_pctr(mu=-4, sigma=1):
    """
    Generate pctr from lognormal distribution with default mu = -4, sigma = 1
    :return:
    """
    return min(lognormal(mu, sigma, 1)[0], 1.0)


def generate_payprice(mu=1.2, sigma=1):
    """
    Generate payprice from normal distribution
    :param mu:
    :param sigma:
    :return:
    """
    return max(0.1, normal(mu, sigma, 1)[0])


if __name__ == '__main__':
    random.seed(0)
    outfile = '/Users/yuanlongchen/Desktop/Bidding_Simulation/src/data/test_bidding_request.csv'
    generate(START_DATE, EPISODES, STEPS, outfile)
