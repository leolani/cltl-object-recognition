import time
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor

import cv2
import jsonpickle
import logging
import numpy as np
import pickle
import requests
from cltl.backend.api.camera import Bounds
from typing import Iterable, Tuple

from cltl.object_recognition.api import Object, ObjectDetector
from cltl.object_recognition.docker import DockerInfra

from PIL import Image
from cltl.object_recognition.plots import Annotator, Colors

ObjectInfo = namedtuple('ObjectInfo', ('type',
                                   'bbox',
                                   'embedding'))


class ObjectDetectorProxy(ObjectDetector):
    def __init__(self):
        self.detect_infra = DockerInfra('tae898/yolov5', 10004, 10004, False, 15)

    def __enter__(self):
        executor = ThreadPoolExecutor(max_workers=2)
        detect = executor.submit(self.detect_infra.__enter__)
        detect.result()
        executor.shutdown()

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        executor = ThreadPoolExecutor(max_workers=2)
        detect = executor.submit(lambda: self.detect_infra.__exit__(exc_type, exc_val, exc_tb))
        detect.result()
        executor.shutdown()

    def detect(self, image: np.ndarray) -> Tuple[Iterable[Object], Iterable[Bounds]]:
        logging.info("Processing image %s", image.shape)

        object_infos = self.detect_objects(image)
        if object_infos:
            objects, bounds = zip(*map(self._to_object, object_infos))
        else:
            objects, bounds = (), ()

        return objects, bounds

    def _to_object(self, object_info: ObjectInfo) -> Tuple[Object, Bounds]:
        bbox = [int(num) for num in object_info.bbox.tolist()]
        representation = object_info.embedding
        type = object_info.type

        return Object(representation, type), Bounds(bbox[0], bbox[2], bbox[1], bbox[3])

    def to_binary_image(self, image: np.ndarray) -> bytes:
        is_success, buffer = cv2.imencode(".png", image)

        if not is_success:
            raise ValueError("Could not encode image")

        return buffer

    def detect_objects(self,
        image: np.ndarray,
        url_object: str = "http://127.0.0.1:10004/"
    ) -> Tuple[ObjectInfo]:
        object_types, object_bboxes, det_scores, landmarks, embeddings = self.run_object_api({"image": self.to_binary_image(image)}, url_object)

        return tuple(ObjectInfo(*info) for info in zip(object_types,
                                              object_bboxes,
                                              embeddings))


    def run_object_api(to_send: dict, url_object: str = "http://127.0.0.1:10004/") -> tuple:
        logging.debug(f"sending image to server...")
        start = time.time()
        to_send = jsonpickle.encode(to_send)
        response = requests.post(url_object, json=to_send)
        logging.info("got %s from server in %s sec", response, time.time()-start)

        response = jsonpickle.decode(response.text)

        object_detection_recognition = response["yolo_results"]

        logging.info(f"{len(object_detection_recognition)} objects detected!")

        object_bboxes = [odr.pop("bbox") for odr in object_detection_recognition]
        det_scores = [odr["det_score"] for odr in object_detection_recognition]
        object_types = [odr["label_string"] for odr in object_detection_recognition]

        embeddings = [fdr["normed_embedding"] for fdr in object_detection_recognition]

        return object_bboxes, det_scores, object_types, embeddings




    def annotate_yolo(image: Image.Image, yolo_results: list) -> Image.Image:
        """Annotate YOLO Image.

        Args
        ----
        image: PIL image object.
        yolo_results: yolo prediction results.

        Returns
        -------
        image_annotated: Annotated PIL image object.

        """
        logging.debug("Annotating yolo image ...")
        annotator = Annotator(np.ascontiguousarray((image)))
        colors = Colors()  # create instance for 'from plots import colors'

        for result in yolo_results:
            box = result["yolo_bbox"]
            label_num = result["label_num"]
            label_string = result["label_string"]

            color = colors(label_num)
            annotator.box_label(box, label_string, color=color)

        image_annotated = Image.fromarray(annotator.im)
        logging.info(f"YOLO image annotation is done!")

        return image_annotated
    