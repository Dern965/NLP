from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import time

def obtener_comentarios(driver, url):
    driver.get(url)
    time.sleep(5)

    comentarios = []
    clases = {'score bueno', 'score regular', 'score malo'}

    while True:
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        tbody = soup.find('tbody')
        if not tbody:
            print(f"No se encontraron comentarios en la página: {url}")
            break

        rows = tbody.find_all('tr')
        profesor_tag = soup.find('h2')
        profesor = profesor_tag.get_text(strip=True) if profesor_tag else "Desconocido"

        for row in rows:
            try:
                fecha = row.find('div', class_='date').get_text(strip=True)
                calificacion = row.find('span', class_='rating-type').get_text(strip=True)

                # Validar la lista antes de acceder a ella
                calidad_general_span = row.find_all('span', class_=clases)
                calidad_general = calidad_general_span[0].get_text(strip=True) if len(calidad_general_span) > 0 else 'N/A'
                facilidad = calidad_general_span[1].get_text(strip=True) if len(calidad_general_span) > 1 else 'N/A'

                clase = row.find('span', class_='name').get_text(strip=True) if row.find('span', class_='name') else 'N/A'
                asistencia = row.find('span', class_='attendance').find('span', class_='response').get_text(strip=True) if row.find('span', class_='attendance') else 'N/A'
                calificacion_recibida = row.find('span', class_='grade').find('span', class_='response').get_text(strip=True) if row.find('span', class_='grade') else 'N/A'
                interes = row.find_all('span', class_='grade')[-1].find('span', class_='response').get_text(strip=True) if row.find_all('span', class_='grade') else 'N/A'
                comentario = row.find('p', class_='commentsParagraph').get_text(strip=True) if row.find('p', class_='commentsParagraph') else 'N/A'
                util = row.find('a', class_='votar_icon helpful').find('span', class_='count').get_text(strip=True) if row.find('a', class_='votar_icon helpful') else '0'
                inutil = row.find('a', class_='votar_icon nothelpful').find('span', class_='count').get_text(strip=True) if row.find('a', class_='votar_icon nothelpful') else '0'

                comentarios.append({
                    'Profesor': profesor,
                    'Fecha': fecha,
                    'Calificación': calificacion,
                    'Calidad General': calidad_general,
                    'Facilidad': facilidad,
                    'Clase': clase,
                    'Asistencia': asistencia,
                    'Calificación Recibida': calificacion_recibida,
                    'Interés en la Clase': interes,
                    'Comentario': comentario,
                    'Útil': util,
                    'Inútil': inutil
                })

            except AttributeError:
                continue

        # Verificar si hay una página siguiente
        next_page = soup.find('a', {'aria-label': 'Siguiente'})
        if next_page and 'href' in next_page.attrs:
            next_url = next_page['href']
            driver.get(next_url)
            time.sleep(5)
        else:
            break

    return comentarios

def main():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-insecure-localhost')
    driver = webdriver.Chrome(service=service, options=options)

    with open('Proyecto/links2.txt', 'r') as file:
        urls = [line.strip() for line in file]
    
    all_comentarios = []
    for url in urls:
        print(f'Procesando: {url}')
        comentarios = obtener_comentarios(driver, url)
        all_comentarios.extend(comentarios)

    # Guardar en CSV
    df = pd.DataFrame(all_comentarios)
    df.to_csv('Comentarios_Prueba.csv', index=False, encoding='utf-8')
    print('Datos guardados en Comentarios_Prueba')

if __name__ == '__main__':
    main()
