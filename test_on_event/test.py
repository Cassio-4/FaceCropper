from module.video_processing_layer import process_event_batch
from module.results_layer import process_batch_result
import config


class TestEvent:
    def __init__(self, name, id, path, total_frames, monitor_id):
        self.__name = name
        self.__id = id
        self.__path = path
        self.__total_frames = total_frames
        self.__monitor_id = monitor_id

    def fspath(self):
        return self.__path

    def name(self):
        return self.__name

    def total_frames(self):
        return self.__total_frames

    def monitor_id(self):
        return self.__monitor_id


class TestMonitor:
    def __init__(self, name, id):
        self.__name = name
        self.__id = id

    def name(self):
        return self.__name

    def id(self):
        return self.__id


def run_test():
    monitor = TestMonitor(config.MONITOR_NAME, config.MONITOR_ID)
    active_monitors = {monitor.id(): monitor}
    event = TestEvent(config.EVENT_NAME, config.EVENT_ID, config.EVENT_PATH, config.TOTAL_FRAMES, config.MONITOR_ID)

    # =============== MAIN LOOP ===============
    event_batch = [event]
    config.LOGGER.Info("Batch has {} events, processing".format(len(event_batch)))
    batch_result, _ = process_event_batch(event_batch)
    config.LOGGER.Info("Batch processing: Done")
    # Process the results of a batch
    process_batch_result(batch_result, active_monitors)


if __name__ == '__main__':
    run_test()
