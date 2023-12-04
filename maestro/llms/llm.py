import base64
import cv2
import numpy as np
import requests

# Constants
META_PROMPT = "For any labels or markings on an image that you reference in your response, please enclose them in square brackets ([]) and list them explicitly. Do not use ranges; for example, instead of '1 - 4', list as '[1], [2], [3], [4]'. These labels could be numbers or letters and typically correspond to specific segments or parts of the image."
OPENAI_API_URL = "https://api.openai.com/v1/chat/completions"

def encode_image_to_base64(image: np.ndarray, format: str = '.jpg') -> str:
    """
    Encodes an image into a base64-encoded string.

    Parameters:
        image (np.ndarray): The image to be encoded.
        format (str): The format to use for encoding ('.jpg' or '.png').

    Returns:
        str: A base64-encoded string representing the image.
    """
    success, buffer = cv2.imencode(format, image)
    if not success:
        raise ValueError(f"Could not encode image to {format} format.")
    return base64.b64encode(buffer).decode('utf-8')

def compose_headers(api_key: str) -> dict:
    """
    Composes the headers needed for an API request.

    Parameters:
        api_key (str): The API key for authenticating requests.

    Returns:
        dict: A dictionary of headers.
    """
    return {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

def compose_openai_payload(image_base64: str, prompt: str) -> dict:
    """
    Composes the payload for a request to the OpenAI API.

    Parameters:
        image_base64 (str): The base64-encoded string of the image.
        prompt (str): The textual prompt to accompany the image.

    Returns:
        dict: A dictionary representing the payload for the API request.
    """
    return {
        "model": "gpt-4-vision-preview",
        "messages": [
            {"role": "system", "content": [META_PROMPT]},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}}
                ]
            }
        ],
        "max_tokens": 800
    }

def prompt_image_local(image: np.ndarray, prompt: str, server_url: str, custom_payload: dict) -> str:
    """
    Sends an image and a textual prompt to a local LLM server.

    Parameters:
        image (np.ndarray): The image to be sent to the API.
        prompt (str): The textual prompt to accompany the image.
        server_url (str): The URL of the local server.
        custom_payload (dict): The custom payload for the local server.

    Returns:
        str: The response from the local API.
    """
    image_base64 = encode_image_to_base64(image)
    payload = custom_payload(image_base64, prompt, META_PROMPT)
    response = requests.post(server_url, headers={"Content-Type": "application/json"}, json=payload).json()
    return response.get("content", "Error-- No content found.")

def prompt_image(api_key: str, image: np.ndarray, prompt: str) -> str:
    """
    Sends an image and a textual prompt to the OpenAI API.

    Parameters:
        api_key (str): The API key.
        image (np.ndarray): The image to be sent to the API.
        prompt (str): The textual prompt to accompany the image.

    Returns:
        str: The response from the OpenAI API.
    """
    image_base64 = encode_image_to_base64(image)
    payload = compose_openai_payload(image_base64, prompt)
    headers = compose_headers(api_key)
    response = requests.post(OPENAI_API_URL, headers=headers, json=payload).json()
    if 'error' in response:
        raise ValueError(response['error']['message'])
    return response['choices'][0]['message']['content']
