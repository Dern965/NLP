from selenium.webdriver import Chrome
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from bs4 import BeautifulSoup
import pandas as pd
import time

def main():
    service = Service(ChromeDriverManager().install())
    options = webdriver.ChromeOptions()
    #options.add_argument('--headless') # No abrir el navegador, osea si estan corriendo en una maquina virtual usen esto
    options.add_argument('--window-size=1920,1080')
    options.add_argument('--ignore-certificate-errors')
    options.add_argument('--allow-insecure-localhost')
    driver = Chrome(service=service, options=options)
    driver.get('https://www.misprofesores.com/escuelas/IPN-ESCOM_1694')
    time.sleep(5) # Esperar a que cargue la pagina y se cierra tras 5s

    # Extraer la tabla de profesores
    html = driver.page_source
    soup = BeautifulSoup(html, 'html.parser')
    tbody = soup.find('tbody')
    rows = tbody.find_all('tr')

    datos = []
    for row in rows:
        cols = row.find_all('td')
        if len(cols) >= 4:
            nombre = cols[1].get_text(strip=True) #Nombre del profesor, el 2 es el nombre el 1 es el enlace
            departamento = cols[3].get_text(strip=True) #Departamento
            calificaciones = cols[4].get_text(strip=True) #Calificaciones
            promedio = cols[5].get_text(strip = True)#Promedio

            datos.append(
                {
                    "Nombre del profesor": nombre,
                    "Departamento": departamento,
                    "# Calificaciones": calificaciones,
                    "Promedio": promedio
                }
            )
    
    driver.quit()

    # Guardar los datos en un archivo CSV
    df = pd.DataFrame(datos)
    df.to_csv('Proyecto\\CSV\Listado_profesores.csv', index=False, encoding='utf-8')
    print('Datos guardados en profesores.csv')

if __name__ == '__main__':
    main()  