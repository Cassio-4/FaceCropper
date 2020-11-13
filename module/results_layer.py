from requests.exceptions import Timeout, ConnectionError
import requests
import base64
import config
import json
import cv2


def encode_image_base64(image, event_name, cropped, num=None):
    # Encode the highest scoring image in base64 to be sent in a JSON request
    # https://stackoverflow.com/questions/40928205/python-opencv-image-to-byte-string-for-json-transfer
    try:
        _, buffer = cv2.imencode('.png', image)
    except cv2.error as e:
        if cropped:
            config.LOGGER.Error("{} PNG encoding error on face crop {{\n".format(event_name) +
                                "    object num = {}\n".format(num) +
                                "    image shape = {}\n}}".format(image.shape))
            cv2.imwrite("output/{}face{}.jpg".format(event_name, num), image)
        elif num is not None:
            config.LOGGER.Error("{} PNG encoding error on trueImage {{\n".format(event_name) +
                                "    object num = {}\n".format(num) +
                                "    image shape = {}\n}}".format(image.shape))
            cv2.imwrite("output/{}trueImage{}.jpg".format(event_name, num), image)
        else:
            config.LOGGER.Error("{} PNG encoding error on alarm frame {{\n".format(event_name) +
                                "    image shape = {}\n}}".format(image.shape))
            cv2.imwrite("output/{}alarm.jpg".format(event_name), image)
        return None
    try:
        b64_image = base64.b64encode(buffer)
        b64_image_with_prefix = "data:image/png;base64," + b64_image.decode("utf-8")
        return b64_image_with_prefix
    except Exception:
        if cropped:
            config.LOGGER.Error("{} Base64 encoding error on face crop: ".format(event_name))
        else:
            config.LOGGER.Error("{} Base64 encoding error on alarm frame: ".format(event_name))
        return None


def send_packet(packet):
    headers = {'content-type': 'application/json; charset=UTF-8'}
    config.LOGGER.log_JSON(packet)
    try:
        response = requests.post('http://localhost/api', data=json.dumps(packet),
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
    mounting the JSONS to be sent via post request

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
            packet["trueImage"] = encode_image_base64(pe.alarmed_frame, pe.event.name(), cropped=False)

            if packet["trueImage"] is None:
                config.LOGGER.Error("{} 's alarm frame encoding error: Callback".format(pe.event.name()))
                continue

            send_packet(packet)

        # If there were detections, flagFace=1 and send image
        else:
            # For each detection in this event
            packet["flagFace"] = 1
            packet["personalType"] = 1
            for num, obj in enumerate(pe.objects):
                # Add to JSON
                packet["trueImage"] = encode_image_base64(obj.highest_detection_frame, pe.event.name(), cropped=False, num=num)
                packet["cropFace"] = encode_image_base64(obj.highest_detection_crop, pe.event.name(), cropped=True, num=num)
                if packet["trueImage"] is None:
                    config.LOGGER.Error("{} error encoding detection number {}'s trueImage".format(pe.event.name(),
                                                                                                   str(num)))
                    callback = True
                if packet["cropFace"] is None:
                    config.LOGGER.Error("{} error encoding detection number {}'s cropFace".format(pe.event.name(),
                                                                                                  str(num)))
                    callback = True
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
