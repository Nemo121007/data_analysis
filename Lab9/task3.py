from bs4 import BeautifulSoup


def get_theaters_with_links(file_path):
    # Открываем HTML-файл
    with open(file_path, "r", encoding="utf-8") as file:
        html_content = file.read()

    # Парсим HTML с помощью BeautifulSoup
    soup = BeautifulSoup(html_content, "html.parser")

    # Находим все ссылки <a> на странице
    links = soup.find_all("a", href=True)

    # Фильтруем ссылки, оставляя только те, что начинаются с /wiki/
    theater_links = {}
    for link in links:
        href = link["href"]
        if href.startswith("/wiki/"):
            theater_name = link.get_text(strip=True)
            if "театр" in theater_name.lower():  # Проверяем, чтобы название содержало слово "театр"
                theater_links[theater_name] = href

    return theater_links


# Пример использования
file_path = "Карелия_Википедия.html"  # Укажите путь к вашему HTML-файлу
theaters = get_theaters_with_links(file_path)

# Получаем wiki-ссылку на "Музыкальный театр Республики Карелия"
result = theaters.get("Музыкальный театр Республики Карелия")
print(f"Ссылка на 'Музыкальный театр Республики Карелия': {result}")
print(result)
