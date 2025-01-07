import pandas as pd
from transformers import pipeline

comentarios = pd.read_csv('Proyecto/CSV/Comentarios_Profesores_Generalizado.csv')
listado = pd.read_csv('Proyecto\CSV\Listado_profesores_Generalizado.csv')
try:
    # modelo_beto = pipeline("text-generation", model="bigscience/bloomz-560m")
    modelo_resumen = pipeline("summarization", model="facebook/bart-large-cnn") 
    # modelo_resumen = pipeline("summarization", model="t5-small")
    print("Modelo cargado exitosamente.")
except Exception as e:
    print("Error cargando el modelo:", e)
    #modelo_beto = None
    modelo_resumen = None

def listado_profesores():
    print("Listado de profesores:")
    #Obtenemos el listado
    profesores_unicos = comentarios['Profesor'].unique()
    for index, profesor in enumerate(profesores_unicos):
        print(f"\t{index + 1}. {profesor}")

def ver_comentarios(profesores_unicos):
    try:
        numero_profesor = int(input('Ingrese el numero del profesor: '))
        if 1 <= numero_profesor <= len(profesores_unicos):
            profesor = profesores_unicos[numero_profesor - 1]
            comentarios_profesor = comentarios[comentarios['Profesor'] == profesor]
            if comentarios_profesor.empty:
                print(f"No hay comentarios disponibles para {profesor}.")
                return
            print(f"Comentarios del profesor {profesor}:")
            for index, comentario in comentarios_profesor.iterrows():
                print(f"\t{index+1}. {comentario['Comentario']}")
        else:
            print('Opcion invalida, intente de nuevo')
    except ValueError:
        print('Opcion invalida, intente de nuevo')


def generar_opinion(profesores_unicos):
    
    try:
        numero_profesor = int(input('Ingrese el numero del profesor: '))
        if 1 <= numero_profesor <= len(profesores_unicos):
            profesor = profesores_unicos[numero_profesor - 1]
            print(f"Generando opinión del profesor {profesor}:")
            comentarios_profesor = comentarios[comentarios['Profesor'] == profesor]["Comentario"]
            if comentarios_profesor.empty:
                print(f"No hay comentarios disponibles para {profesor}.")
                return

            comentarios_profesor = comentarios_profesor.dropna()  
            comentarios_profesor = comentarios_profesor.astype(str) 

            if len(comentarios_profesor) <= 4:  
                print(f"El profesor {profesor} tiene {len(comentarios_profesor)} comentarios. Mostrando todos los comentarios disponibles:")
                for comentario in comentarios_profesor:
                    print(f"- {comentario}")
                return
            
            elif 5 <= len(comentarios_profesor) < 9:
                texto_base = ' '.join(comentarios_profesor.tolist())
            else:    
                texto_base = ' '.join(comentarios_profesor.sample(n=9, random_state=37).tolist())
            
            print(f"Texto base para generación: {texto_base}")

            if modelo_resumen is None:
                print("El modelo de resumen no está disponible. Verifica su carga.")
                return
            
            if not texto_base.strip():
                print("No hay suficiente texto base para generar un resumen.")
                return
            
            try:

                """
                opinion_generada = modelo_beto(
                    texto_base, max_new_tokens=50, max_length = 50,num_return_sequences=1, truncation=True
                )[0]['generated_text']
                print(f"Opinion generada: {opinion_generada}")
                
                opinion_generada = modelo_resumen(
                    texto_base, max_length=300, min_length=110,truncation=True
                )[0]['summary_text']
                print(f"Resumen generado: {opinion_generada}")
                """
                # El de arriba es el modelo de t5

                opinion_generada = modelo_resumen(
                    texto_base, max_length=300, min_length=150, do_sample=False
                )[0]['summary_text']
                print("\n############################################\n")
                print(f"Resumen generado: {opinion_generada}")

            except Exception as e:
                print("Error generando la opinión:", e)
        else:
            print('Opción inválida, intente de nuevo 1')
    except ValueError:
        print('Opción inválida, intente de nuevo 2')

def sub_menu():
    profesores_unicos = comentarios['Profesor'].unique()
    while True:
        print('MENU')
        print('1. Ver comentarios')
        print('2. Generar opinion')
        print('3. Salir')

        try:
            opcion = int(input('Ingrese una opcion: '))
            if opcion == 1:
                ver_comentarios(profesores_unicos)
            elif opcion == 2: 
                generar_opinion(profesores_unicos)
            elif opcion == 3:
                break
            else:
                print('Opcion invalida, intente de nuevo')
        except ValueError:
            print('Opcion invalida, intente de nuevo')
if __name__ == '__main__':
    profesores_unicos = comentarios['Profesor'].unique()
    while True:
        print('MENU')
        print('1. Listado de profesores')
        print('2. Salir')

        try:
            opcion = int(input('Ingrese una opcion: '))
            if opcion == 1:
                listado_profesores()
                sub_menu()
            elif opcion == 2:
                break
            else:
                print('Opcion invalida, intente de nuevo')
        except ValueError:
            print('Opcion invalida, intente de nuevo')