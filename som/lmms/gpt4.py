import base64

import cv2
import numpy as np
import requests

META_PROMPT = '''
For any labels or markings on an image that you reference in your response, please 
enclose them in square brackets ([]) and list them explicitly. Do not use ranges; for 
example, instead of '1 - 4', list as '[1], [2], [3], [4]'. These labels could be 
numbers or letters and typically correspond to specific segments or parts of the image.
'''

API_URL = "https://api.openai.com/v1/chat/completions"


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    Encodes an image into a base64-encoded string in JPEG format.

    Parameters:
        image (np.ndarray): The image to be encoded. This should be a numpy array as
            typically used in OpenCV.

    Returns:
        str: A base64-encoded string representing the image in JPEG format.
    """
    success, buffer = cv2.imencode('.jpg', image)
    if not success:
        raise ValueError("Could not encode image to JPEG format.")

    encoded_image = base64.b64encode(buffer).decode('utf-8')
    return encoded_image


def compose_headers(api_key: str) -> dict:
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }


def compose_payload(image: np.ndarray, prompt: str) -> dict:
    base64_image = encode_image_to_base64(image)
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
                "role": "system",
                "content": [
                    META_PROMPT
                ]
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 800
    }


def prompt_image(api_key: str, image: np.ndarray, prompt: str) -> str:
    """
    Sends an image and a textual prompt to the OpenAI API and returns the API's textual
    response.

    This function integrates an image with a user-defined prompt to generate a response
    using OpenAI's API.

    Parameters:
        api_key (str): The API key for authenticating requests to the OpenAI API.
        image (np.ndarray): The image to be sent to the API.
            used in OpenCV.
        prompt (str): The textual prompt to accompany the image in the API request.

    Returns:
        str: The textual response from the OpenAI API based on the input image and
            prompt.

    Raises:
        ValueError: If there is an error in encoding the image or if the API response
            contains an error.
    """
    headers = compose_headers(api_key=api_key)
    payload = compose_payload(image=image, prompt=prompt)
    response = requests.post(url=API_URL, headers=headers, json=payload).json()

    if 'error' in response:
        raise ValueError(response['error']['message'])
    return response['choices'][0]['message']['content']
