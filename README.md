
<div align="center">

  <h1>multimodal-maestro</h1>

  <br>

  [![version](https://badge.fury.io/py/multimodal-maestro.svg)](https://badge.fury.io/py/multimodal-maestro)
  [![license](https://img.shields.io/pypi/l/multimodal-maestro)](https://github.com/roboflow/multimodal-maestro/blob/main/LICENSE)
  [![python-version](https://img.shields.io/pypi/pyversions/multimodal-maestro)](https://badge.fury.io/py/multimodal-maestro)
  [![Gradio](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Roboflow/SoM)
  [![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/roboflow/multimodal-maestro/blob/main/cookbooks/multimodal_maestro_gpt_4_vision.ipynb
)

</div>

## 👋 hello

Multimodal-Maestro gives you more control over large multimodal models to get the 
outputs you want. With more effective prompting tactics, you can get multimodal models 
to do tasks you didn't know (or think!) were possible. Curious how it works? Try our 
HF [space](https://huggingface.co/spaces/Roboflow/SoM)!

🚧 The project is still under construction and the API is prone to change.

## 💻 install

Pip install the supervision package in a
[**3.11>=Python>=3.8**](https://www.python.org/) environment.

```bash
pip install multimodal-maestro
```

## 🚀 examples

### GPT-4 Vision

```
Find dog.

>>> The dog is prominently featured in the center of the image with the label [9].
```

<details close>
<summary>👉 read more</summary>

<br>

- **load image**

  ```python
  import cv2
  
  image = cv2.imread("...")
  ```

- **create and refine marks**

  ```python
  import multimodalmaestro as mm
  
  generator = mm.SegmentAnythingMarkGenerator(device='cuda')
  marks = generator.generate(image=image)
  marks = mm.refine_marks(marks=marks)
  ```

- **visualize marks**

  ```python
  mark_visualizer = mm.MarkVisualizer()
  marked_image = mark_visualizer.visualize(image=image, marks=marks)
  ```
  ![image-vs-marked-image](https://github.com/roboflow/multimodal-maestro/assets/26109316/92951ed2-65c0-475a-9279-6fd344757092)

- **prompt**

  ```python
  prompt = "Find dog."
  
  response = mm.prompt_image(api_key=api_key, image=marked_image, prompt=prompt)
  ```
  
  ```
  >>> "The dog is prominently featured in the center of the image with the label [9]."
  ```

- **extract related marks**

  ```python
  masks = mm.extract_relevant_masks(text=response, detections=refined_marks)
  ```
  
  ```
  >>> {'6': array([
  ...     [False, False, False, ..., False, False, False],
  ...     [False, False, False, ..., False, False, False],
  ...     [False, False, False, ..., False, False, False],
  ...     ...,
  ...     [ True,  True,  True, ..., False, False, False],
  ...     [ True,  True,  True, ..., False, False, False],
  ...     [ True,  True,  True, ..., False, False, False]])
  ... }
  ```

</details>

![multimodal-maestro](https://github.com/roboflow/multimodal-maestro/assets/26109316/c04f2b18-2a1d-4535-9582-e5d3ec0a926e)

## 🚧 roadmap

- [ ] Documentation page.
- [ ] Segment Anything guided marks generation.
- [ ] Non-Max Suppression marks refinement.
- [ ] LLaVA demo.

## 💜 acknowledgement

- [Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding
in GPT-4V](https://arxiv.org/abs/2310.11441) by Jianwei Yang, Hao Zhang, Feng Li, Xueyan
Zou, Chunyuan Li, Jianfeng Gao.

## 🦸 contribution

We would love your help in making this repository even better! If you noticed any bug, 
or if you have any suggestions for improvement, feel free to open an 
[issue](https://github.com/roboflow/set-of-mark/issues) or submit a 
[pull request](https://github.com/roboflow/set-of-mark/pulls).
