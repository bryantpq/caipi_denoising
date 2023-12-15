import datetime
import logging
import os
import socket
import sys

def create_logger(config_name, logging_level):
    hostname = socket.gethostname()
    if 'titan' in hostname:
        log_folder = '/home/quahb/caipi_denoising/logs'
    elif hostname in ['compbio', 'hpc']:
        log_folder = '/common/quahb/caipi_denoising/logs'
    else:
        raise ValueError(f'Unkown hostname: {hostname}')

    date = datetime.date.today()
    fname = os.path.join(log_folder, config_name + '_{}.log'.format(date))

    if logging_level.lower() == 'info':
        level = logging.INFO
    else:
        level = logging.DEBUG

    logging.basicConfig(
            level=level,
            format='%(asctime)s %(filename)20s:%(lineno)d [%(levelname)s]: %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            handlers=[
                logging.FileHandler(fname),
                logging.StreamHandler(sys.stdout)
            ]
    )
