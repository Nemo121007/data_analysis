import requests


def get_headers():
    # Выполняем GET-запрос
    response = requests.get("https://httpbin.org/get")
    # Преобразуем текст ответа в JSON
    response_json = response.json()
    # Возвращаем значение заголовка "Host" из секции "headers"
    return response_json["headers"]["Host"]


# Проверяем работу функции
print(get_headers())  # Должно вернуть "httpbin.org"
