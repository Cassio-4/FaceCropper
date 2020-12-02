from requests.exceptions import Timeout, ConnectionError
import requests
import base64
import config
import json
from json import JSONDecodeError
import cv2


def encode_image_base64(image):
    # Encode the highest scoring image in base64 to be sent in a JSON request
    # https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
    try:
        _, buffer = cv2.imencode('.png', image)
        b64_image = base64.b64encode(buffer)
        b64_image_with_prefix = "data:image/png;base64," + b64_image.decode("utf-8")
    except cv2.error as e:
        config.LOGGER.Error("PNG CV2 encoding error {{\n" +
                            "    image shape = {}\n}}".format(image))
        return None
    except Exception:
        config.LOGGER.Error("Base64 encoding error.")
        return None

    return b64_image_with_prefix


def send_packet(packet, pe):
    config.LOGGER.log_JSON(packet)
    try:
        response = requests.post('http://localhost/api', data=json.dumps(packet),
                                 headers={'content-type': 'application/json; charset=UTF-8'}, verify=False)
        config.LOGGER.log_JSON(response.content.decode('utf-8'))
        config.LOGGER.Info("ATENTO status: {}".format(int(response.json()["status"])))

        pe.set_keep_video(True if response.json()["video"] == "true" else False)

    except Timeout:
        config.LOGGER.Error("Request Timed Out.")
        pe.set_callback(True)
    except ConnectionError:
        config.LOGGER.Error("Connection error.")
        pe.set_callback(True)
    except JSONDecodeError:
        config.LOGGER.Error("Json Decode error.")


def process_batch_result(batch_result, monitors):
    """
    Processes the results from a batch of events, being responsible for
    mounting the JSONS to be sent via post request

    :param batch_result: a list of ProcessedEvent objects
    :param monitors: a dict of monitors
    :return:
    """
    # For processed event in a batch result
    for pe in batch_result:
        config.LOGGER.Info("Creating JSON for {}.".format(pe.event.name()))
        # Grab this event's monitor name
        monitor_name = monitors.get(pe.event.monitor_id()).name()
        # Fill the common parts of the request
        packet = {
            "requestNumber": 00,
            "companyCode": 4,
            "dispositiveType": 2,
            "captureDeviceCode": monitor_name,
            "appCode": 7,
            "latitude": "null",
            "longitude": "null",
            "truePictureTree": "99",
            "eventName": pe.event.name()
        }

        # If there were no detections, we will send an alarmed frame of this event
        if len(pe.objects) == 0:
            packet["flagFace"] = 0
            packet["trueImage"] = encode_image_base64(pe.alarmed_frame)

            if packet["trueImage"] is None:
                config.LOGGER.Error("{} 's alarm frame encoding error: Callback".format(pe.event.name()))
                pe.set_callback(True)
                continue

            send_packet(packet, pe)

        # If there were detections
        else:
            # For each detection in this event
            packet["flagFace"] = 1
            packet["personalType"] = 1
            for num, obj in enumerate(pe.objects):
                # Add to JSON
                packet["trueImage"] = encode_image_base64(obj.highest_detection_frame)
                packet["cropFace"] = encode_image_base64(obj.highest_detection_crop)
                if packet["trueImage"] is None:
                    config.LOGGER.Error("{} error encoding detection number {}'s trueImage".format(pe.event.name(),
                                                                                                   str(num)))
                    pe.set_callback(True)
                    continue
                if packet["cropFace"] is None:
                    config.LOGGER.Error("{} error encoding detection number {}'s cropFace".format(pe.event.name(),
                                                                                                  str(num)))
                    pe.set_callback(True)
                    continue

                # Send it to ATENTO API
                send_packet(packet, pe)

        config.LOGGER.Info("Creating JSON for {}: Done.".format(pe.event.name()))
