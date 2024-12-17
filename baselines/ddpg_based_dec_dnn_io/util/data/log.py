import logging
import os
import sys
import time

def ensure_dir(file_path: str):
    """Create it if the directory of :attr:`file_path` is not existed.

    Args:
        file_path (str): Target file path.
    """

    if not os.path.isdir(file_path):
        file_path = os.path.dirname(file_path)

    if not os.path.exists(file_path):
        print(file_path)
        os.makedirs(file_path)

def get_cur_time_str():
    """Get the current timestamp string like '20210618123423' which contains date and time information.

    Returns:
        str: Current timestamp string.
    """
    return time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))

logger = logging.getLogger('zedl')
logger.setLevel(logging.DEBUG)
logger.propagate = False

formatter = logging.Formatter("%(asctime)s - %(filename)s[%(lineno)d] - %(levelname)s: %(message)s")
log_dir_path = os.path.join(os.path.dirname(os.path.abspath(sys.argv[0])), './log')
if not os.path.exists(log_dir_path):
    os.mkdir(log_dir_path)

# file log
cur_time_str = get_cur_time_str()
log_file_path = os.path.join(log_dir_path, cur_time_str[0:8], cur_time_str[8:] + '.log')
ensure_dir(log_file_path)
file_handler = logging.FileHandler(log_file_path, mode='a')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# cmd log
stream_handler = logging.StreamHandler()
stream_handler.setLevel(logging.INFO)
stream_handler.setFormatter(formatter)
logger.addHandler(stream_handler)

logging.getLogger('nni').setLevel(logging.ERROR)

# copy file content to log file
with open(os.path.abspath(sys.argv[0]), 'r', encoding='utf-8') as f:
    content = f.read()
    logger.debug('entry file content: ---------------------------------')
    logger.debug('\n' + content)
    logger.debug('entry file content: ---------------------------------')


