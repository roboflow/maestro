
# CHANGELOGS

## multimodal-maestro-0.1.0

### 🚀 Added

- [`SegmentAnythingMarkGenerator`](https://roboflow.github.io/multimodal-maestro/markers/#multimodalmaestro.markers.sam.SegmentAnythingMarkGenerator) allowing the generation of segmentation marks.
- [`MarkVisualizer`](https://roboflow.github.io/multimodal-maestro/visualizers/#multimodalmaestro.visualizers.MarkVisualizer) allowing to visualize the generated marks.
- [`prompt_image`](https://roboflow.github.io/multimodal-maestro/lmms/#multimodalmaestro.lmms.gpt4.prompt_image) allowing for convenient GPT-4 Vision API querying.
- 🤗 Hugging Face Set-of-Mark [space](https://huggingface.co/spaces/Roboflow/SoM).

```python
>>> import cv2
>>> import torch
>>> import multimodalmaesto as mm

>>> image = cv2.imread("...")

>>> generator = mm.SegmentAnythingMarkGenerator()
>>> visualizer = mm.MarkVisualizer()

>>> marks = generator.generate(image=image)
>>> marks = mm.refine_marks(marks=marks)

>>> image_prompt = visualizer.visualize(image=image, marks=marks)
>>> text_prompt = "Find dog."

>>> response = mm.prompt_image(api_key=api_key, image=image_prompt, prompt=text_prompt)
>>> response

"The dog is prominently featured in the center of the image with the label [9]."

>>> masks = mm.extract_relevant_masks(text=response, detections=refined_marks)

{'6': array([
    [False, False, False, ..., False, False, False],
    [False, False, False, ..., False, False, False],
    [False, False, False, ..., False, False, False],
    ...,
    [ True,  True,  True, ..., False, False, False],
    [ True,  True,  True, ..., False, False, False],
    [ True,  True,  True, ..., False, False, False]])
}
```

![multimodal-maestro-2](https://github.com/roboflow/multimodal-maestro/assets/26109316/118feb2e-654e-473c-b534-65bc01df7480)
