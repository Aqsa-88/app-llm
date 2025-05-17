import requests

response = requests.post(
    url="https://localhost:8000/generate",
    json = {"prompt ": "Hello, world!","max_lenght":80}
)

print(response.json())




















