from utils.pyzm_wrappers import Logger
from utils.cache_wrapper import EventDeleterCache

logger = Logger()
ev = EventDeleterCache(maxsize=3, logger=logger, delete=True)
ev[1] = 'a'
ev[2] = 'b'
ev[3] = 'c'
ev[4] = 'd'
ev[5] = '3'
