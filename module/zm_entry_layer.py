from module.video_processing_layer import process_event_batch
from module.results_layer import process_batch_result
from utils.cache_wrapper import EventDeleterCache
import datetime
import schedule
import config
import time


def retrieve_monitors(zmapi):
    active_monitors = {}
    passive_monitors = {}

    config.LOGGER.Info("Getting monitors")
    zm_monitors = zmapi.monitors()
    for monitor in zm_monitors.list():
        if monitor.id() in config.PASSIVE_MONITORS:
            config.LOGGER.Debug(1, 'Monitor: {} [id: {}] is a passive monitor.'.format(monitor.name(), monitor.id()))
            passive_monitors[monitor.id()] = monitor
        else:
            config.LOGGER.Debug(1, 'Monitor: {} [id: {}] is an active monitor.'.format(monitor.name(), monitor.id()))
            active_monitors[monitor.id()] = monitor
    config.LOGGER.Info("Getting monitors: Done")
    return active_monitors, passive_monitors


def get_events_for_monitors(monitors, event_filter, events_cache):
    event_batch = []
    # Get all events and create a batch
    for monitor in monitors.values():
        config.LOGGER.Debug(1, 'Getting events for {}.'.format(monitor.name()))
        cam_events = monitor.events(options=event_filter)
        for event in cam_events.list():
            if event.id() not in events_cache:
                event_batch.append(event)
                config.LOGGER.Debug(1, 'Event:{} Cause:{} Notes:{}'.format(event.name(),
                                                                           event.cause(), event.notes()))
    # Return the batch of events
    return event_batch


def process_passive_monitors(zmapi):
    config.LOGGER.Info("It's late, processing passive monitors")
    event_filter = {
        'from': "24 hours ago",
        'object_only': False,
        'min_alarmed_frames': 1
    }

    _, passive_monitors = retrieve_monitors(zmapi)
    # If there are no passive monitors then there is nothing to be done
    if len(passive_monitors) == 0:
        return
    # Get all events for the passive monitors
    event_batch = get_events_for_monitors(passive_monitors, event_filter, None)
    # Send batch to be processed
    config.LOGGER.Info("Batch has {} events, processing".format(len(event_batch)))
    batch_result = process_event_batch(event_batch)
    config.LOGGER.Info("Batch processing: Done")
    process_batch_result(batch_result, passive_monitors)
    config.LOGGER.Info("Processing passive monitors: Done")


def login_with_api():
    from getpass import getpass
    import pyzm.api as zmapi
    import traceback

    password = getpass("Please enter your password to access your Zoneminder server: ")
    api_options = {
        'apiurl': config.API_URL,
        'portalurl': config.PORTAL_URL,
        'user': config.USER,
        'password': password,
        'logger': config.LOGGER
    }
    if config.DISABLE_SSL:
        api_options['disable_ssl_cert_check'] = True

    config.LOGGER.Info('Running FaceCropper on {}'.format(api_options['apiurl']))

    # lets init the API
    try:
        zmapi = zmapi.ZMApi(options=api_options)
    except Exception as e:
        print('Error: {}'.format(str(e)))
        print(traceback.format_exc())
        exit(1)

    return zmapi


def update_event_filter(event_filter):
    now = datetime.datetime.now()
    ago = now - datetime.timedelta(hours=5)
    event_filter['from'] = "{}-{}-{} {}:{}:{}".format(ago.year, ago.month, ago.day, ago.hour, ago.minute,
                                                      ago.second)
    event_filter['to'] = "{}-{}-{} {}:{}:{}".format(now.year, now.month, now.day, now.hour, now.minute,
                                                    now.second)


def run_module():
    zmapi = login_with_api()

    # Get active monitors
    active_monitors, _ = retrieve_monitors(zmapi)

    # Before entering the main loop, schedule the passive monitors
    schedule.every().day.at("00:00").do(process_passive_monitors, zmapi)

    # Starting the events filter
    event_filter = {
        'from': None,
        'to': None,
        'object_only': False,
        'min_alarmed_frames': 1
    }

    # This is a cache to remember what events were received in the previous calls
    events_cache = EventDeleterCache(maxsize=10, logger=config.LOGGER, delete=config.DELETE_PROCESSED_EVENTS)
    # =============== MAIN LOOP ===============
    while True:
        config.LOGGER.Info("Pooling for events")
        time.sleep(config.POOLING_TIME)

        update_event_filter(event_filter)

        config.LOGGER.Info("Getting events for active monitors")
        event_batch = get_events_for_monitors(active_monitors, event_filter, events_cache)
        # Sleeping for n seconds in case the events received are still being written
        time.sleep(3.0)
        # Send batch to be processed
        config.LOGGER.Info("Batch has {} events, processing".format(len(event_batch)))
        batch_result, _ = process_event_batch(event_batch)
        config.LOGGER.Info("Batch processing: Done")
        # Process the results of a batch
        process_batch_result(batch_result, active_monitors)

        for pe in batch_result:
            events_cache[pe.event.id()] = pe.event
