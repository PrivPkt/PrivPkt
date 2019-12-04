
import logging


def get_logger(name):
    FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(format=FORMAT)
    res = logging.getLogger(name)
    res.setLevel(logging.DEBUG)
    return res

