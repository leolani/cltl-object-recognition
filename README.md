# cltl-object-recognition

This repository is a component of the [Leolani framework](https://github.com/leolani/cltl-combot).
For usage of the component within the framework see the instructions there.

## Object recognition (cltl.object_recognition)

The component provides Object recognition on images.

### API

The component API provides an `ObjectDetector` that accepts an image and returns a list of detected `Objects` and a
list of their bounding boxes:

    objects, bounds = object_detector.detect(image)

### Implementations

#### Yolo5 object recognition

_cltl.object_recognition.proxy_ provides an implementation that uses a Dockerized version of
[Yolo5 object recognition](https://github.com/ultralytics/yolov5), provided by
[https://github.com/tae898/yolov5](https://github.com/tae898/yolov5).

##### Configuration

The implementation can be configured in the section

    [cltl.object_recognition]
    start_infra: Falses
    detector_url: http://object-recognition:10004/

* _start_infra_: Start the Docker image in the _ObjectDetectorProxy_
* _detector_url_: If _start_infra_ in set to _False_, connect to the provided URL

## Integration (cltl_service.object_recognition)


### Events

The service for the component accepts events that carry an _ImageSignal_ as payload and for each
received event it emits an event on the output topic that carries a list of _Mentions_, annotating
bounding boxes in the image with the detected objects.

Example output event:
```json
{
  "mentions": [
    {
      "annotations": [
        {
          "source": "python-source:cltl.object_recognition#0.0.1",
          "timestamp": 1663166417633,
          "type": "python-type:cltl.object_recognition.api.Object",
          "value": {
            "type": "chair"
          }
        }
      ],
      "id": "d8428d8d-20ec-4e04-9491-6c79eb905f66",
      "segment": [
        {
          "bounds": [0, 0, 1, 1],
          "container_id": "36729510-236f-414b-839c-d43f8d93d2c8"
        }
      ]
    }
  ],
  "type": "ObjectRecognitionEvent"
}
```

### Topic Configuration

For integration the input and output topics must be defined in the following configuration section:

    [cltl.object_recognition.events]
    image_topic: <input topic name>
    object_topic: <output topic name>

## Contributing

Contributions are what make the open source community such an amazing place to be learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

<!-- LICENSE -->

## License

Distributed under the MIT License. See [`LICENSE`](https://github.com/leolani/cltl-combot/blob/main/LICENCE) for more
information.

<!-- CONTACT -->

## Authors

* [Taewoon Kim](https://tae898.github.io/)
* [Thomas Baier](https://www.linkedin.com/in/thomas-baier-05519030/)
* [Selene Báez Santamaría](https://selbaez.github.io/)
* [Piek Vossen](https://github.com/piekvossen)
