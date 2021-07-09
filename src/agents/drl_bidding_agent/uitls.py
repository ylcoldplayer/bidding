from datetime import datetime, timedelta
import yaml

DATE_FORMAT = '%Y-%m-%dT%H::%M::%S.%f'


def get_timestamp_from_obs(obs):
    return obs['timestamp']


def string_to_datetime(timestamp):
    return datetime.strptime(timestamp, DATE_FORMAT)


def datetime_to_str(dt):
    return dt.strftime(DATE_FORMAT)


def step_diff(t1, t2, interval=15):
    """

    :param t1:
    :param t2:
    :param interval: step duration in minutes
    :return:
    """
    duration = string_to_datetime(t2) - string_to_datetime(t1)
    delta = timedelta(minutes=interval)
    return int(duration.total_seconds()/delta.total_seconds())


def increment_ts(ts, interval):
    """

    :param ts:
    :param interval:
    :return:
    """
    ts_dt = string_to_datetime(ts)
    delta = timedelta(minutes=interval)
    ts_dt_new = ts_dt + delta
    return datetime_to_str(ts_dt_new)


def increment_ts_by_one_day(ts):
    """
    Incremental ts by one day(1440 minutes)
    :param ts:
    :return:
    """
    return increment_ts(ts, 1440)


def same_date(ts1, ts2):
    """

    :param ts1:
    :param ts2:
    :return:
    """
    dt1 = string_to_datetime(ts1)
    dt2 = string_to_datetime(ts2)
    return dt1.date() == dt2.date()


def load_yaml(yaml_file):
    """

    :param yaml_file:
    :return:
    """
    with open(yaml_file, 'r') as f:
        return yaml.load(f, Loader=yaml.FullLoader)


def get_start_date(config_file):
    """

    :param config_file:
    :return:
    """
    yaml_dict = load_yaml(config_file)
    return yaml_dict['start_date']


def get_T(config_file):
    """

    :param config_file:
    :return:
    """
    yaml_dict = load_yaml(config_file)
    T = 96
    if 'T' in yaml_dict:
        T = yaml_dict['T']
    return T


def get_total_budget(config_file):
    """

    :param config_file:
    :return:
    """
    yaml_dict = load_yaml(config_file)
    return yaml_dict['total_budget']


def get_target_value(config_file):
    """

    :param config_file:
    :return:
    """
    yaml_dict = load_yaml(config_file)
    return yaml_dict['target_value']


def get_dqn_state_size(config_file):
    """

    :param config_file:
    :return:
    """
    yaml_dict = load_yaml(config_file)
    return yaml_dict['dqn_state_size']


def get_dqn_action_size(config_file):
    """

    :param config_file:
    :return:
    """
    yaml_dict = load_yaml(config_file)
    return yaml_dict['dqn_action_size']


if __name__ == '__main__':
    ts_test_1 = '2021-07-04T15::11::45.456777'
    ts_test_2 = '2021-07-04T15::50::45.456777'
    interval = 15
    print(step_diff(ts_test_1, ts_test_2, interval))
    print(increment_ts(ts_test_1, 15))

    ts_test_3 = '2021-07-05T15::50::45.456777'
    print(same_date(ts_test_1, ts_test_2))
    print(same_date(ts_test_1, ts_test_3))

    yaml_file = './config.yaml'
    start_date = get_start_date(yaml_file)
    print(type(start_date))
    print(type(get_T(yaml_file)))
    print(increment_ts_by_one_day(start_date))
    print(get_target_value(yaml_file))
    print(get_total_budget(yaml_file))
    print(get_dqn_action_size(yaml_file))
    print(get_dqn_state_size(yaml_file))
