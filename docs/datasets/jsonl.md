## Overview

JSONL dataset is a simple, text-based format that makes it easy to work with multimodal data. Each line in a
JSONL file is a valid JSON object. Each JSON object must contain the following keys:

- `image`: A string specifying the image file name associated with the dataset item.
- `prefix`: A string representing the prompt that will be sent to the model.
- `suffix`: A string representing the expected model response.

!!! warning

    `suffix` can be as simple as a single number or string, a full paragraph, or even a structured JSON output.
    Regardless of its content, ensure that the it is properly serialized to conform with JSON value requirements.

!!! tip

    Use Roboflow's tools to [annotate and export](https://blog.roboflow.com/multimodal-dataset-labeling/) your
    multimodal datasets in JSONL format, streamlining data preparation for model training.

## Dataset Structure

Divide your dataset into three subdirectories: `train`, `valid`, and `test`. Each subdirectory should contain its own
`annotations.jsonl` file that holds the annotations for that particular split, along with the corresponding image
files. Below is an example of the directory structure:

```text
dataset/
├── train/
│   ├── annotations.jsonl
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
├── valid/
│   ├── annotations.jsonl
│   ├── image1.jpg
│   ├── image2.jpg
│   └── ... (other image files)
└── test/
    ├── annotations.jsonl
    ├── image1.jpg
    ├── image2.jpg
    └── ... (other image files)
```

## JSONL Examples

JSONL is a versatile format that can represent datasets for a wide range of visual-language tasks. Its flexible
structure supports multiple annotation styles, making it an ideal choice for integrating diverse data sources.

### Object Character Recognition (OCR)

OCR extracts textual content from images, converting printed or handwritten text into machine-readable data.

```text
{"image":"image1.jpg","prefix":"read equation in LATEX","suffix":"H = \\dot { x } _ { i } \\Pi _ { x ^ { i } } + \\Pi _ { x ^ { i } } \\dot { x } _ { i } ^ { * } + \\dot { \\psi } _ { i } \\Pi _ { \\psi _ { i } } - \\Pi _ { \\psi _ { i } ^ { * } } \\dot { \\psi } _ { i } ^ { * } +"}
{"image":"image2.jpg","prefix":"read equation in LATEX","suffix":"\\psi _ { j } ( C _ { r } ^ { \\vee } , t ) = \\frac { 4 \\sinh 2 j t ( \\cosh ( 2 w _ { 1 } t ) \\cosh ( 2 w _ { 2 } t ) - \\cos ^ { 2 } ( x t ) ) } { \\sinh 2 t \\cosh h t } ."}
{"image":"image3.jpg","prefix":"read equation in LATEX","suffix":"- \\frac { h ^ { 2 } } { 2 \\lambda } \\int d t d ^ { 2 } x d ^ { 2 } x ^ { \\prime } ( { \\tilde { J } } _ { k } - \\frac { J _ { k } ^ { 0 } } { \\rho _ { 0 } } { \\tilde { J } } _ { 0 } ) ( t , x ) \\Delta ^ { - 1 } ( x - x ^ { \\prime } ) ( { \\tilde { J } } _ { k } - \\frac { J _ { k } ^ { 0 } } { \\rho _ { 0 } } { \\tilde { J } } _ { 0 } ) ( t , x ^ { \\prime } ) ."}
```

<table style="width: 100%;">
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/9559367b-30b4-4673-91f4-bc3dfe11124c" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         read equation in LATEX</p>
      <p><strong>suffix:</strong><br>
         H = \\dot { x } _ { i } \\Pi _ { x ^ { i } } + \\Pi _ { x ^ { i } } \\dot { x } _ { i } ^ { * } + \\dot { \\psi } _ { i } \\Pi _ { \\psi _ { i } } - \\Pi _ { \\psi _ { i } ^ { * } } \\dot { \\psi } _ { i } ^ { * } +</p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/c2e5fc81-c5d8-48d6-bfa9-fc97f4d40b3a" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         read equation in LATEX</p>
      <p><strong>suffix:</strong><br>
         \\psi _ { j } ( C _ { r } ^ { \\vee } , t ) = \\frac { 4 \\sinh 2 j t ( \\cosh ( 2 w _ { 1 } t ) \\cosh ( 2 w _ { 2 } t ) - \\cos ^ { 2 } ( x t ) ) } { \\sinh 2 t \\cosh h t } .</p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/59b055f5-9724-45fd-986d-fd261c9cc9f5" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         read equation in LATEX</p>
      <p><strong>suffix:</strong><br>
         - \\frac { h ^ { 2 } } { 2 \\lambda } \\int d t d ^ { 2 } x d ^ { 2 } x ^ { \\prime } ( { \\tilde { J } } _ { k } - \\frac { J _ { k } ^ { 0 } } { \\rho _ { 0 } } { \\tilde { J } } _ { 0 } ) ( t , x ) \\Delta ^ { - 1 } ( x - x ^ { \\prime } ) ( { \\tilde { J } } _ { k } - \\frac { J _ { k } ^ { 0 } } { \\rho _ { 0 } } { \\tilde { J } } _ { 0 } ) ( t , x ^ { \\prime } ) .</p>
    </td>
  </tr>
</table>

### Visual Question Answering (VQA)

VQA tasks require models to answer natural language questions based on the visual content of an image.

```text
{"image":"image1.jpg","prefix":"What is the ratio of yes to no?","suffix":"1.54"}
{"image":"image2.jpg","prefix":"What was the leading men's magazine in the UK from April 2019 to March 2020?","suffix":"GQ"}
{"image":"image3.jpg","prefix":"Which country has the greatest increase from 1975 to 1980?","suffix":"Taiwan"}
```

<table style="width: 100%;">
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/c8f72d98-9b0b-4730-806b-bb810f70b19d" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         What is the ratio of yes to no?</p>
      <p><strong>suffix:</strong><br>
         1.54</p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/7d130d44-d8b8-4a45-9a22-d69e30d76e26" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         What was the leading men's magazine in the UK from April 2019 to March 2020?</p>
      <p><strong>suffix:</strong><br>
         GQ</p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/d811d1c4-d05f-497d-87de-efd79d75ee23" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         Which country has the greatest increase from 1975 to 1980?</p>
      <p><strong>suffix:</strong><br>
         Taiwan</p>
    </td>
  </tr>
</table>

### JSON Data Extraction

This task involves identifying and extracting structured information formatted as JSON from images or documents,
facilitating seamless integration into data pipelines.

```text
{"image":"image1.jpg","prefix":"extract document data in JSON format","suffix":"{\"route\": \"J414-YG-624\",\"pallet_number\": \"17\",\"delivery_date\": \"9/18/2024\",\"load\": \"1\",\"dock\": \"D08\",\"shipment_id\": \"P18941494362\",\"destination\": \"595 Navarro Radial Suite 559, Port Erika, HI 29655\",\"asn_number\": \"4690787672\",\"salesman\": \"CAROL FREDERICK\",\"products\": [{\"description\": \"159753 - BOX OF PAPER CUPS\",\"cases\": \"32\",\"sales_units\": \"8\",\"layers\": \"2\"},{\"description\": \"583947 - BOX OF CLOTH RAGS\",\"cases\": \"8\",\"sales_units\": \"2\",\"layers\": \"5\"},{\"description\": \"357951 - 6PK OF HAND SANITIZER\",\"cases\": \"2\",\"sales_units\": \"32\",\"layers\": \"4\"},{\"description\": \"847295 - CASE OF DISPOSABLE CAPS\",\"cases\": \"16\",\"sales_units\": \"4\",\"layers\": \"3\"}],\"total_cases\": \"58\",\"total_units\": \"46\",\"total_layers\": \"14\",\"printed_date\": \"12/05/2024 10:14\",\"page_number\": \"60\"}"}
{"image":"image2.jpg","prefix":"extract document data in JSON format","suffix":"{\"route\": \"V183-RZ-924\",\"pallet_number\": \"14\",\"delivery_date\": \"5/3/2024\",\"load\": \"4\",\"dock\": \"D20\",\"shipment_id\": \"P29812736099\",\"destination\": \"706 Meghan Brooks, Amyberg, IA 67863\",\"asn_number\": \"2211190904\",\"salesman\": \"RYAN GREEN\",\"products\": [{\"description\": \"293847 - ROLL OF METAL WIRE\",\"cases\": \"16\",\"sales_units\": \"8\",\"layers\": \"4\"},{\"description\": \"958273 - CASE OF SPRAY MOPS\",\"cases\": \"16\",\"sales_units\": \"8\",\"layers\": \"3\"},{\"description\": \"258963 - CASE OF MULTI-SURFACE SPRAY\",\"cases\": \"2\",\"sales_units\": \"4\",\"layers\": \"2\"}],\"total_cases\": \"34\",\"total_units\": \"20\",\"total_layers\": \"9\",\"printed_date\": \"12/05/2024 10:14\",\"page_number\": \"91\"}"}
{"image":"image3.jpg","prefix":"extract document data in JSON format","suffix":"{\"route\": \"A702-SG-978\",\"pallet_number\": \"19\",\"delivery_date\": \"4/7/2024\",\"load\": \"5\",\"dock\": \"D30\",\"shipment_id\": \"Y69465838537\",\"destination\": \"31976 French Wall, East Kimport, NY 87074\",\"asn_number\": \"4432967070\",\"salesman\": \"PATRICIA ROSS\",\"products\": [{\"description\": \"384756 - CASE OF BUCKET LIDS\",\"cases\": \"32\",\"sales_units\": \"4\",\"layers\": \"3\"},{\"description\": \"384756 - CASE OF BUCKET LIDS\",\"cases\": \"8\",\"sales_units\": \"32\",\"layers\": \"4\"},{\"description\": \"958273 - CASE OF SPRAY MOPS\",\"cases\": \"32\",\"sales_units\": \"2\",\"layers\": \"5\"},{\"description\": \"345678 - BOX OF DISPOSABLE GLOVES\",\"cases\": \"64\",\"sales_units\": \"16\",\"layers\": \"3\"}],\"total_cases\": \"136\",\"total_units\": \"54\",\"total_layers\": \"15\",\"printed_date\": \"11/29/2024 17:03\",\"page_number\": \"28\"}"}
```

<table style="width: 100%;">
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/0e5552dd-83f0-444b-b930-787ae733e5ca" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         extract document data in JSON format</p>
      <p><strong>suffix:</strong><br>
        {"route": "J414-YG-624","pallet_number": "17","delivery_date": "9/18/2024","load": "1","dock": "D08","shipment_id": "P18941494362","destination": "595 Navarro Radial Suite 559, Port Erika, HI 29655","asn_number": "4690787672","salesman": "CAROL FREDERICK","products": [{"description": "159753 - BOX OF PAPER CUPS","cases": "32","sales_units": "8","layers": "2"},{"description": "583947 - BOX OF CLOTH RAGS","cases": "8","sales_units": "2","layers": "5"},{"description": "357951 - 6PK OF HAND SANITIZER","cases": "2","sales_units": "32","layers": "4"},{"description": "847295 - CASE OF DISPOSABLE CAPS","cases": "16","sales_units": "4","layers": "3"}],"total_cases": "58","total_units": "46","total_layers": "14","printed_date": "12/05/2024 10:14","page_number": "60"}</p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/3a4da3c8-2379-4799-9e4e-79a2c2ce1513" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         extract document data in JSON format</p>
      <p><strong>suffix:</strong><br>
         {"route": "V183-RZ-924","pallet_number": "14","delivery_date": "5/3/2024","load": "4","dock": "D20","shipment_id": "P29812736099","destination": "706 Meghan Brooks, Amyberg, IA 67863","asn_number": "2211190904","salesman": "RYAN GREEN","products": [{"description": "293847 - ROLL OF METAL WIRE","cases": "16","sales_units": "8","layers": "4"},{"description": "958273 - CASE OF SPRAY MOPS","cases": "16","sales_units": "8","layers": "3"},{"description": "258963 - CASE OF MULTI-SURFACE SPRAY","cases": "2","sales_units": "4","layers": "2"}],"total_cases": "34","total_units": "20","total_layers": "9","printed_date": "12/05/2024 10:14","page_number": "91"}</p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/d583a919-11db-4403-9dd3-441fcad999ba" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         extract document data in JSON format</p>
      <p><strong>suffix:</strong><br>
         {"route": "A702-SG-978","pallet_number": "19","delivery_date": "4/7/2024","load": "5","dock": "D30","shipment_id": "Y69465838537","destination": "31976 French Wall, East Kimport, NY 87074","asn_number": "4432967070","salesman": "PATRICIA ROSS","products": [{"description": "384756 - CASE OF BUCKET LIDS","cases": "32","sales_units": "4","layers": "3"},{"description": "384756 - CASE OF BUCKET LIDS","cases": "8","sales_units": "32","layers": "4"},{"description": "958273 - CASE OF SPRAY MOPS","cases": "32","sales_units": "2","layers": "5"},{"description": "345678 - BOX OF DISPOSABLE GLOVES","cases": "64","sales_units": "16","layers": "3"}],"total_cases": "136","total_units": "54","total_layers": "15","printed_date": "11/29/2024 17:03","page_number": "28"}</p>
    </td>
  </tr>
</table>

### Object Detection

This task involves detecting and localizing multiple objects within an image by drawing bounding boxes around them.
Each Vision-Language Model (VLM) may require a different text representation of these bounding boxes to interpret the
spatial data correctly. The annotations below are compatible with PaliGemma and PaliGemma 2.

!!! tip

    We are rolling out support for COCO and YOLO formats soon, and will handle conversion between bounding box
    representations and the format required by each supported VLM.

```text
{"image":"image1.jpg","prefix":"detect figure ; table ; text","suffix":"<loc0412><loc0102><loc0734><loc0920> figure ; <loc0744><loc0102><loc0861><loc0920> text ; <loc0246><loc0102><loc0404><loc0920> text ; <loc0085><loc0102><loc0244><loc0920> text"}
{"image":"image2.jpg","prefix":"detect figure ; table ; text","suffix":"<loc0516><loc0114><loc0945><loc0502> text ; <loc0084><loc0116><loc0497><loc0906> figure ; <loc0517><loc0518><loc0945><loc0907> text"}
{"image":"image3.jpg","prefix":"detect figure ; table ; text","suffix":"<loc0784><loc0174><loc0936><loc0848> text ; <loc0538><loc0174><loc0679><loc0848> table ; <loc0280><loc0177><loc0533><loc0847> figure ; <loc0068><loc0174><loc0278><loc0848> figure ; <loc0686><loc0174><loc0775><loc0848> text"}
```

<table style="width: 100%;">
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/42106f5d-6c44-418c-8e57-68bbc91c2f82" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         detect figure ; table ; text</p>
      <p><strong>suffix:</strong><br>
         &lt;loc0412&gt;&lt;loc0102&gt;&lt;loc0734&gt;&lt;loc0920&gt; figure ;
         &lt;loc0744&gt;&lt;loc0102&gt;&lt;loc0861&gt;&lt;loc0920&gt; text ;
         &lt;loc0246&gt;&lt;loc0102&gt;&lt;loc0404&gt;&lt;loc0920&gt; text ;
         &lt;loc0085&gt;&lt;loc0102&gt;&lt;loc0244&gt;&lt;loc0920&gt; text
      </p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/0210f8da-1bf3-45bd-80a2-6d8d1e20ab3e" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         detect figure ; table ; text</p>
      <p><strong>suffix:</strong><br>
         &lt;loc0516&gt;&lt;loc0114&gt;&lt;loc0945&gt;&lt;loc0502&gt; text ;
         &lt;loc0084&gt;&lt;loc0116&gt;&lt;loc0497&gt;&lt;loc0906&gt; figure ;
         &lt;loc0517&gt;&lt;loc0518&gt;&lt;loc0945&gt;&lt;loc0907&gt; text
      </p>
    </td>
  </tr>
  <tr>
    <td style="width: 40%;">
      <img src="https://github.com/user-attachments/assets/d7df56e0-0fad-42ca-b8c9-535d04c0fe11" style="width: 100%; height: auto;" alt="Image">
    </td>
    <td style="width: 60%; text-align: left;">
      <p><strong>prefix:</strong><br>
         detect figure ; table ; text</p>
      <p><strong>suffix:</strong><br>
         &lt;loc0784&gt;&lt;loc0174&gt;&lt;loc0936&gt;&lt;loc0848&gt; text ;
         &lt;loc0538&gt;&lt;loc0174&gt;&lt;loc0679&gt;&lt;loc0848&gt; table ;
         &lt;loc0280&gt;&lt;loc0177&gt;&lt;loc0533&gt;&lt;loc0847&gt; figure ;
         &lt;loc0068&gt;&lt;loc0174&gt;&lt;loc0278&gt;&lt;loc0848&gt; figure ;
         &lt;loc0686&gt;&lt;loc0174&gt;&lt;loc0775&gt;&lt;loc0848&gt; text
      </p>
    </td>
  </tr>
</table>
