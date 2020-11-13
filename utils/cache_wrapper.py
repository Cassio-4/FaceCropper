from cachetools import LRUCache
from requests import HTTPError


class EventDeleterCache(LRUCache):
    def __init__(self, maxsize, logger, delete):
        super(EventDeleterCache, self).__init__(maxsize)
        self.logger = logger
        self.delete = delete

    def popitem(self):
        key, value = super().popitem()
        # Value should be a zm event object
        try:
            if self.delete:
                value.delete()
                self.logger.Info("Cache deleted ZM event: {}".format(key))
            else:
                self.logger.Debug(1, "Cache full: releasing {}".format(key))
        except HTTPError as e:
            self.logger.Error("Failed to delete {} on ZM: ".format(value.name()))

