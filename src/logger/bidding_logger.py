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


def get_bidding_logger(logger_name):
    """
    Logger for bidding
    :return:
    """
    logging.config.dictConfig(log_config)
    bidding_logger = logging.getLogger(logger_name)
    return bidding_logger


if __name__ == '__main__':
    logger = get_bidding_logger('bidding')
    logger.info('test')
