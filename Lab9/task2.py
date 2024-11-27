from bs4 import BeautifulSoup


def count_div_a_tags(file_path):
    # Открываем HTML-файл
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Парсим HTML с помощью BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Считаем количество тегов <div> и <a>
    div_count = len(soup.find_all("div"))
    a_count = len(soup.find_all("a"))

    # Суммируем количество тегов
    sum_da = div_count + a_count

    return sum_da


# Пример использования
file_path = "ПетрГУ_Википедия.html"  # Укажите путь к вашему HTML-файлу
result = count_div_a_tags(file_path)
print(f"Сумма тегов <div> и <a>: {result}")
