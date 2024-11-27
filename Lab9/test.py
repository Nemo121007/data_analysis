import requests
import base64

response = requests.get('https://httpbin.org/get')  # Записываем ответ на наш запрос в переменную response
print('Объект Response:', response)
print('Код ответа:', response.status_code)

print('Запрос успешен?', response.ok)
print('Содержимое ответа:')
print(response.content)
print('\nТекст ответа (строка):')
print(response.text)

print('Кодировка ответа (используется при распознавании .content в .text):', response.encoding)
print('Заголовки ответа (массив):', response.headers)

parsed_json = response.json()  # Получение ответа в формате json
print('Результат .json() - набор данных в стандартных структурах python:', type(parsed_json))

print('Отправленный заголовок User-Agent:', parsed_json['headers']['User-Agent'])

response = requests.get(
    'https://httpbin.org/get',
    params={'My Param': 'My value', '?!&': '*&@!', 'array': [1, 2, 3]},  # Задаем параметры
    headers={'X-My-Header': 'SomeValue'},  # Задаем заголовки
    auth=('Login', 'Password'),  # Передаем данные для авторизации
    cookies={'My Cookie': 'Cookie Value'}  # Передаем cookie (если это не первый запрос к адресу)
)
print(response.text)  # Печатаем ответ

# Вычленяем из ответа связку логин:пароль и преобразуем в тип данных bytes
base64.b64decode(response.json()['headers']['Authorization'].split()[1])

response = requests.post(
    'https://httpbin.org/post',  # Адрес запроса
    data='Data in post request body',  # Данные запроса
    params={'Data in': 'url parameters'}  # Можно посылать независимо от data-post
)
print(response.text)

response = requests.post(
    'https://httpbin.org/post',
    data={'field1': 'Hello', 'field2': 'World'}
)
print(response.text)

response = requests.post(
    'https://httpbin.org/post',
    json={'field1': 'Hello', 'field2': 'World'},  # Передаем данные в формате json
)
print(response.text)

# Создаем файл для отправки
with open('request_test.txt', 'w') as f:
  f.write('some data')

response = requests.post(
    'https://httpbin.org/post',
    files={'request_test.txt': ('request_test.txt', open('request_test.txt'))}
)
print(response.text)

