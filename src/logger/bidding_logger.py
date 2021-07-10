import logging
import logging.config

log_config = {
    "version": 1,
    "root": {
        "handlers": ["console", "file"],
        "level": "DEBUG"
    },
    "handlers": {
        "console": {
            "formatter": "std_out",
            "class": "logging.StreamHandler",
            "level": "DEBUG"
        },
        "file": {
            "formatter": "std_out",
            "class": "logging.FileHandler",
            "level": "INFO",
            "filename": "bidding.log"
        }
    },
    "formatters": {
        "std_out": {
            "format": "%(levelname)s : %(module)s : %(funcName)s : %(message)s",
        }
    },
}


def get_bidding_logger(logger_name, file_name=None):
    """
    Logger for bidding
    :return:
    """
    config_dict = log_config
    if file_name:
        config_dict['handlers']['file']['filename'] = file_name
    logging.config.dictConfig(config_dict)
    bidding_logger = logging.getLogger(logger_name)
    return bidding_logger


if __name__ == '__main__':
    logger = get_bidding_logger('drl_bidding')
    logger.info('test')
