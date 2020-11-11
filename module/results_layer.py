from requests.exceptions import Timeout, ConnectionError
import requests
import base64
import config
import json
import cv2


def encode_image_base64(image):
    # Encode the highest scoring image in base64 to be sent in a JSON request
    # https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
    try:
        _, buffer = cv2.imencode('.png', image)
        b64_image = base64.b64encode(buffer)
        b64_image_with_prefix = "data:image/png;base64," + b64_image.decode("utf-8")
        return b64_image_with_prefix
    except cv2.error as e:
        config.LOGGER.Error(e)
        return None


def send_packet(packet):
    headers = {'content-type': 'application/json; charset=UTF-8'}
    config.LOGGER.log_JSON(packet)
    try:
        response = requests.post('https://www.atento.inf.br/api', data=json.dumps(packet),
                                 headers=headers, verify=False)
    except Timeout:
        config.LOGGER.Error("Request Timed Out.")
    except ConnectionError:
        config.LOGGER.Error("Connection error.")
    else:
        config.LOGGER.log_JSON(response.content.decode('utf-8'))


def process_batch_result(batch_result, monitors):
    """
    Processes the results from a batch of events, being responsible for
    mounting and sending info to ATENTO

    :param batch_result: a list of ProcessedEvent objects
    :param monitors: a dict of monitors
    :return:
    """
    # For processed event in a batch result
    for pe in batch_result:
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
                config.LOGGER.Error("unable to encode image")
                continue
            send_packet(packet)
        # If there were detections, flagFace = 1 and send image
        else:
            # For each detection in this event
            packet["flagFace"] = 1
            packet["personalType"] = 1
            for obj in pe.objects:
                # Add to JSON
                packet["trueImage"] = encode_image_base64(obj.highest_detection_frame)
                packet["cropFace"] = encode_image_base64(obj.highest_detection_crop)
                if packet["trueImage"] is None:
                    config.LOGGER.Error("unable to encode image")
                    continue
                # packet["atributesPerson"] = [{"atribute": "Age", "value": 31},
                #                            {"atribute": "Color", "value": 1},
                #                            {"atribute": "Sex", "value": "1"},
                #                            {"atribute": "Direction", "value": 2}]

                # Send it to ATENTO API
                send_packet(packet)

                if config.SAVE_DETECTIONS:
                    cv2.imwrite("output/{}{}.jpg".format(pe.event.name(), obj.id), obj.highest_detection_crop)
                    cv2.imwrite("output/trueImage{}{}.jpg".format(pe.event.name(), obj.id), obj.highest_detection_frame)
