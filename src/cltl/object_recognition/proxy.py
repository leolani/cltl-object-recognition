import logging
import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, Tuple

import cv2
import jsonpickle
import numpy as np
import requests
from cltl.backend.api.camera import Bounds

from cltl.object_recognition.api import Object, ObjectDetector
from cltl.combot.infra.docker import DockerInfra

logger = logging.getLogger(__name__)


ObjectInfo = namedtuple('ObjectInfo', ('type', 'bbox'))


class ObjectDetectorProxy(ObjectDetector):
    def __init__(self, start_infra: bool = True, detector_url: str = None):
        if start_infra:
            self._detect_infra = DockerInfra('tae898/yolov5', 10004, 10004, False, 15)
            self._detector_url = "http://127.0.0.1:10004/"
        else:
            self._detect_infra = None
            self._detector_url = detector_url

        if not self._detector_url:
            raise ValueError("No url defined for docker image")

    def __enter__(self):
        if self._detect_infra:
            executor = ThreadPoolExecutor(max_workers=2)
            detect = executor.submit(self._detect_infra.__enter__)
            detect.result()
            executor.shutdown()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._detect_infra:
            executor = ThreadPoolExecutor(max_workers=2)
            detect = executor.submit(lambda: self._detect_infra.__exit__(exc_type, exc_val, exc_tb))
            detect.result()
            executor.shutdown()

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Object], Iterable[Bounds]]:
        logger.info("Processing image %s", image.shape)

        object_infos = self.detect_objects(image)
        if object_infos:
            objects, bounds = zip(*map(self._to_object, object_infos))
        else:
            objects, bounds = (), ()

        return objects, bounds

    def _to_object(self, object_info: ObjectInfo) -> Tuple[Object, Bounds]:
        bbox = [int(num) for num in object_info.bbox]
        type = object_info.type

        return Object(type), Bounds(bbox[0], bbox[2], bbox[1], bbox[3])

    def to_binary_image(self, image: np.ndarray) -> bytes:
        is_success, buffer = cv2.imencode(".png", image)

        if not is_success:
            raise ValueError("Could not encode image")

        return buffer

    def detect_objects(self, image: np.ndarray) -> Tuple[ObjectInfo]:
        object_bboxes, det_scores, object_types = self.run_object_api({"image": self.to_binary_image(image)})

        return tuple(ObjectInfo(*info) for info in zip(object_types, object_bboxes))

    def run_object_api(self, to_send: dict) -> tuple:
        logger.debug(f"sending image to server...")
        start = time.time()
        to_send = jsonpickle.encode(to_send)
        response = requests.post(self._detector_url, json=to_send)

        logger.info("got %s from server in %s sec", response, time.time()-start)

        response = jsonpickle.decode(response.text)
        object_detection_recognition = response["yolo_results"]

        logger.debug("%s objects detected!", len(object_detection_recognition))

        object_bboxes = [odr.pop("bbox") for odr in object_detection_recognition]
        det_scores = [odr["det_score"] for odr in object_detection_recognition]
        object_types = [odr["label_string"] for odr in object_detection_recognition]

        logger.info("Detected %s objects: %s", len(object_detection_recognition), object_types)

        return object_bboxes, det_scores, object_types
