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

#### VLM-based object recognition (Ollama)

_cltl.object_recognition.ollama_proxy_ provides an alternative implementation, `OllamaObjectDetectorProxy`,
that detects objects by prompting a vision-language model (e.g. [Qwen2.5-VL](https://ollama.com/library/qwen2.5vl)
or [Qwen3-VL](https://ollama.com/library/qwen3-vl)) served through [Ollama](https://ollama.com), instead of
relying on a dedicated object detection model. The model is asked to return detections as structured JSON with
normalized bounding boxes, which are converted to the same `Object`/`Bounds` shape used by the rest of the
component, so it can be used as a drop-in replacement for `ObjectDetectorProxy`.

Reasoning ("thinking") models are requested to answer without emitting their chain of thought
(`think: false`), and the response is parsed defensively in case a model ignores that and leaks
`<think>...</think>` reasoning into its answer anyway (a known issue for some Qwen3-VL tags, see
[ollama/ollama#14798](https://github.com/ollama/ollama/issues/14798)). To avoid this entirely, prefer a
non-thinking, instruction-tuned tag such as `qwen3-vl:4b-instruct` or `qwen3-vl:8b-instruct` over a bare
`qwen3-vl:<size>` or `-thinking` tag.

In addition to individual objects, the model is asked to classify the overall scene or place depicted in
the image, e.g. `office`, `kitchen`, `living room`, `street`, `city`, `village`, `forest`. This is returned
as an additional `Object` with `type` set to `"scene"`, whose `Bounds` cover the complete image, so it can
be distinguished from the localized object detections (which have `type` set to the model name).

Ollama can either be run locally, or accessed through Ollama's cloud API, which can run larger models
(e.g. Qwen) without needing local GPU hardware.

##### Configuration

The implementation can be configured in the section

    [cltl.object_recognition.ollama]
    model: qwen2.5vl
    host: https://ollama.com

* _model_: Name of the (vision-capable) Ollama model to use. Defaults to `qwen2.5vl`.
* _host_: Address of the Ollama server. If omitted, falls back to the `OLLAMA_HOST` environment
  variable, or a local Ollama instance (`http://127.0.0.1:11434`). Set to `https://ollama.com` to use
  Ollama's cloud API.
* _api_key_: API key for Ollama's cloud API. For security, prefer setting the `OLLAMA_API_KEY`
  environment variable over storing it in the config file. Not needed for a local instance.

Detection quality and the reliability of bounding boxes depend on the chosen model, and are generally
less precise than a dedicated object detector like Yolo5.

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
