# recommender_system_test
Proyecto A/B

# Introducción
Has recibido una tarea analítica de una tienda en línea internacional. Tus predecesores no consiguieron completarla: 
lanzaron una prueba A/B y luego abandonaron (para iniciar una granja de sandías en Brasil). Solo dejaron las especificaciones técnicas y los resultados de las pruebas.

# Metodología
**Preprocesamiento de datos:** Se limpiaron y estandarizaron los datos, eliminando inconsistencias y verificando la ausencia de duplicados y valores faltantes.
**Exploratory Data Analysis (EDA):** Se analizaron distintos datos, se crearon graficas para visualizacion de datos mas claramente
**Pruebas A/B:** Se realizaron pruebas A/B para comprobar si el comportambiento entre ambos grupos es significativamente distinto.

Herramientas utilizadas en este proyecto:

![Python](https://img.shields.io/badge/python-357ebd?style=for-the-badge&logo=python&logoColor=white)
![Pandas](https://img.shields.io/badge/pandas-%23357ebd.svg?style=for-the-badge&logo=pandas&logoColor=white)
![Seaborn](https://img.shields.io/badge/Seaborn-357ebd?style=for-the-badge)
![scikit-learn](https://img.shields.io/badge/scikit--learn-%23357ebd.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Limpieza de datos](https://img.shields.io/badge/Limpieza_de_datos-295F98?style=for-the-badge)
![Transformación de datos](https://img.shields.io/badge/Transformación_de_datos-295F98?style=for-the-badge)
![Análisis de datos](https://img.shields.io/badge/Análisis_de_datos-295F98?style=for-the-badge)
![Pruebas A/B](https://img.shields.io/badge/A/B_Testing-orange)


# Objetivo:
### - ¿Qué queremos conseguir con esta información?
El objetivo de este analisis es evaluar la efectividad de las pruebas A/B realizadas por el equipo anterior, 
para asi determinar si hubo diferencias significativas en el comportamiento de los usuarios entre los grupos A y B en las pruebas que se mencionan:

**recommender_system_test**
**interface_eu_test**

Tambien se debe analizar la influencia de campañas de marketing al probar cambios relacionados con la introducción de un sistema de recomendaciones 
mejorada y evaluar la distribución geográfica, dispositivo utilizado y actividad para detectar posibles sesgos en la asignación de grupos.


# Conclusiones generales y recomendaciones:
## En base al análisis:
Deacuerdo a la información previa , podemos observar un gran desequilibrio de datos entre el Grupo A y B de la prueba recommender_system_test. 
Mientras que en interface_eu_test muestra una distribución bastante equilibrada entre los grupos A y B. Por lo cual podemos continuar con la prueba A/B con este modelo.
Para recommender_system_test el desequilibrio puede sesgar los resultados, por lo cual se recomienda reequilibrar los grupos o recolectar más datos para el grupo B antes de analizar los resultados. 
En el caso de los eventos, solo el 11.2% de los eventos ocurrieron durante el periodo navideño, Esto es muy importante y se debe considerar ya que el comportamiento del usuario puede cambiar significativamente durante las vacaciones.

# Diccionario de datos:

Para acceder a los datasets de la plataforma, agrega /datasets/ al principio de la ruta del archivo (por ejemplo, /datasets/ab_project_marketing_events_us.csv).

ab_project_marketing_events_us.csv — el calendario de eventos de marketing para 2020
final_ab_new_users_upd_us.csv — todos los usuarios que se registraron en la tienda en línea desde el 7 hasta el 21 de diciembre de 2020
final_ab_events_upd_us.csv — todos los eventos de los nuevos usuarios en el período comprendido entre el 7 de diciembre de 2020 y el 1 de enero de 2021
final_ab_participants_upd_us.csv — tabla con los datos de los participantes de la prueba
Estructura ab_project__marketing_events_us.csv:

name — el nombre del evento de marketing
regions — regiones donde se llevará a cabo la campaña publicitaria
start_dt — fecha de inicio de la campaña
finish_dt — fecha de finalización de la campaña
Estructura final_ab_new_users_upd_us.csv:

user_id
first_date — fecha de inscripción
region
device — dispositivo utilizado para la inscripción
Estructura final_ab_events_upd_us.csv:

user_id
event_dt — fecha y hora del evento
event_name — nombre del tipo de evento
details — datos adicionales sobre el evento (por ejemplo, el pedido total en USD para los eventos purchase)
Estructura final_ab_participants_upd_us.csv:

user_id
ab_test — nombre de la prueba
group — el grupo de prueba al que pertenecía el usuario
