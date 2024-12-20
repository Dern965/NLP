from selenium.webdriver import Chrome
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import time

def obtener_links(driver):
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    tbody = soup.find('tbody')
    rows = tbody.find_all('tr')

    links = []
    for row in rows:
        cols = row.find_all('td')
        if cols and cols[1].find('a'):
            link = cols[1].find('a')['href']
            links.append(link)
    return links

def main():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    # options.add_argument('--headless') # Descomentar para no abrir el navegador
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-insecure-localhost')
    driver = webdriver.Chrome(service=service, options=options)

    # URL PRINCIPAL
    base_url = 'https://www.misprofesores.com/escuelas/IPN-ESCOM_1694'
    driver.get(base_url)
    time.sleep(5)

    links_profesores = obtener_links(driver)

    links_profesores = obtener_links(driver)
    if links_profesores:
        links_profesores.pop(0) # Eliminar el primer link que no es de un profesor

    with open('Proyecto\\links.txt', 'w') as f:
        for link in links_profesores:
            f.write(link + '\n')
    print(f'Se encontraron {len(links_profesores)} profesores\n')

    driver.quit()

if __name__ == '__main__':
    main()