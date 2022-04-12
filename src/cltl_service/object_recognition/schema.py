import uuid
from cltl.backend.api.camera import Bounds
from cltl.combot.event.emissor import AnnotationEvent
from cltl.combot.infra.time_util import timestamp_now
from dataclasses import dataclass
from emissor.representation.scenario import Mention, ImageSignal, Annotation
from typing import Iterable

from cltl.object_recognition.api import Object


@dataclass
class ObjectRecognitionEvent(AnnotationEvent[Annotation[Object]]):
    @classmethod
    def create(cls, image_signal: ImageSignal, objects: Iterable[Object], bounds: Iterable[Bounds]):
        if objects:
            mentions = [ObjectRecognitionEvent.to_mention(image_signal, object, bound)
                        for object, bound in zip(objects, bounds)]
        else:
            mentions = [ObjectRecognitionEvent.to_mention(image_signal)]

        return cls(cls.__name__, mentions)

    @staticmethod
    def to_mention(image_signal: ImageSignal, object: Object = None, bounds: Bounds = None):
        segment = image_signal.ruler
        if bounds:
            clipped = Bounds(segment.bounds[0], segment.bounds[2],
                             segment.bounds[1], segment.bounds[3]).intersection(bounds)
            segment = segment.get_area_bounding_box(clipped.x0, clipped.y0, clipped.x1, clipped.y1)

        annotation = Annotation(Object.__name__, object, __name__, timestamp_now())

        return Mention(str(uuid.uuid4()), [segment], [annotation])