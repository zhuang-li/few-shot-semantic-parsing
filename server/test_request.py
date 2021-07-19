import requests
import urllib.parse

url = 'http://localhost:8081/parse/english'
utterance = input()
print(requests.get(url, params={'q': utterance}).json())
