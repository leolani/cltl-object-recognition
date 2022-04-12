import logging

import unittest
from cltl.backend.api.camera import CameraResolution
from cltl.backend.source.cv2_source import SystemImageSource

from cltl.face_recognition.proxy import FaceDetectorProxy


import matplotlib.pyplot as plt

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s.%(msecs)03d %(levelname)s %(module)s - %(funcName)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)


class TestFaceDetectorProxy(unittest.TestCase):
    def test_face_proxy(self):
        with FaceDetectorProxy() as proxy:
            with SystemImageSource(CameraResolution.QQVGA) as source:
                image = source.capture().image
                logging.info("Captured image: %s", image.shape)
                plt.imshow(image)
                plt.show()

            faces, bounds = proxy.detect(image)
            print(faces)
            print(bounds)
