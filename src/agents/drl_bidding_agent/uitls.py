from datetime import datetime, timedelta

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


if __name__ == '__main__':
    ts_test_1 = '2021-07-04T15::11::45.456777'
    ts_test_2 = '2021-07-04T15::50::45.456777'
    interval = 15
    print(step_diff(ts_test_1, ts_test_2, interval))
    print(increment_ts(ts_test_1, 15))

    ts_test_3 = '2021-07-05T15::50::45.456777'
    print(same_date(ts_test_1, ts_test_2))
    print(same_date(ts_test_1, ts_test_3))
