from cachetools import TTLCache


class EventDeleterCache(TTLCache):
    def __init__(self, maxsize, ttl, logger, delete):
        super(EventDeleterCache, self).__init__(maxsize, ttl)
        self.logger = logger
        self.delete = delete

    def popitem(self):
        key, value = super().popitem()
        # Value should be a zm event object
        if self.delete:
            value.delete()
            self.logger.Info("Cache deleted event: {}".format(key))
        else:
            self.logger.Info("Event {} expired from cache".format(key))
