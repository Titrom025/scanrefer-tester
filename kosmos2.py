import base64
import os
import requests


invoke_url = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/functions/0bcd1a8c-451f-4b12-b7f0-64b4781190d1"
fetch_url_format = "https://api.nvcf.nvidia.com/v2/nvcf/pexec/status/"

# Reading auth info from "kosmos_auth.txt" file with following content:
# "Bearer nvapi-XXXXXXXXXXX"
with open("kosmos_auth.txt") as kosmos_auth_file:
   kosmos_auth = kosmos_auth_file.read().strip()

headers = {
    "Authorization": kosmos_auth,
    "Accept": "application/json",
}

def call_kosmos(image_path, prompt):
  with open(image_path, "rb") as image_file:
      image_data = image_file.read()

  image_format = os.path.splitext(image_path)[1][1:]
  base64_encoded_image = base64.b64encode(image_data).decode("utf-8")

  data_uri = f"data:image/{image_format};base64,{base64_encoded_image}"

  payload = {
    "messages": [
      {
        "content": f"{prompt} <img src=\"{data_uri}\" />",
        "role": "user"
      }
    ],
    "bounding_boxes": True,
    "temperature": 0.2,
    "top_p": 0.7,
    "max_tokens": 1024
  }

  session = requests.Session()

  response = session.post(invoke_url, headers=headers, json=payload)

  while response.status_code == 202:
      request_id = response.headers.get("NVCF-REQID")
      fetch_url = fetch_url_format + request_id
      response = session.get(fetch_url, headers=headers)

  response.raise_for_status()
  response_body = response.json()
  response_text = response_body['choices'][0]['message']['content']

  response_text = response_text.replace("</phrase>", "").replace("<phrase>", "")

  return response_text
