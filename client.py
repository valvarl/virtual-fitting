import json
import requests

url = "http://localhost:8000/preproc_pair"

# Подготовка данных для отправки

data = {'stream': False}
files = [('files', open('assets/00013_00.jpg', 'rb')), ('files', open('assets/00017_00.jpg', 'rb'))]

# Отправка POST-запроса
response = requests.post(url, 
                         data=data, 
                         files=files, stream=True)

if response.status_code == 200:
    for line in response.iter_lines():
        if line:
            json_data = json.loads(line)
            # Здесь можно обрабатывать каждый JSON-объект
            print(json_data)

