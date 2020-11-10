from cachetools import LRUCache


class EventDeleterCache(LRUCache):
    def __init__(self, maxsize, logger, delete):
        super(EventDeleterCache, self).__init__(maxsize)
        self.logger = logger
        self.delete = delete

    def popitem(self):
        key, value = super().popitem()
        # Value should be a zm event object
        if self.delete:
            value.delete()
            self.logger.Info("Cache deleted event: {}".format(key))
        else:
            self.logger.Debug(1, "Cache full: {} deleted".format(key))
