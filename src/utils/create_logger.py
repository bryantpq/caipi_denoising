import datetime
import logging
import os
import sys

def create_logger(config_name):
    log_folder = '/home/quahb/caipi_denoising/logs'
    date = datetime.date.today()
    fname = os.path.join(log_folder, config_name + '_{}.log'.format(date))

    logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(filename)20s:%(lineno)d [%(levelname)s]: %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            handlers=[
                logging.FileHandler(fname),
                logging.StreamHandler(sys.stdout)
            ]
    )
