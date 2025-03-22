#!/usr/bin/env python
# coding: utf-8

# Hola **Rafael**!
# 
# Soy **Patricio Requena** 👋. Es un placer ser el revisor de tu proyecto el día de hoy!
# 
# Revisaré tu proyecto detenidamente con el objetivo de ayudarte a mejorar y perfeccionar tus habilidades. Durante mi revisión, identificaré áreas donde puedas hacer mejoras en tu código, señalando específicamente qué y cómo podrías ajustar para optimizar el rendimiento y la claridad de tu proyecto. Además, es importante para mí destacar los aspectos que has manejado excepcionalmente bien. Reconocer tus fortalezas te ayudará a entender qué técnicas y métodos están funcionando a tu favor y cómo puedes aplicarlos en futuras tareas. 
# 
# _**Recuerda que al final de este notebook encontrarás un comentario general de mi parte**_, empecemos!
# 
# Encontrarás mis comentarios dentro de cajas verdes, amarillas o rojas, ⚠️ **por favor, no muevas, modifiques o borres mis comentarios** ⚠️:
# 
# 
# <div class="alert alert-block alert-success">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si todo está perfecto.
# </div>
# 
# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si tu código está bien pero se puede mejorar o hay algún detalle que le hace falta.
# </div>
# 
# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor</b> <a class=“tocSkip”></a>
# Si de pronto hace falta algo o existe algún problema con tu código o conclusiones.
# </div>
# 
# Puedes responderme de esta forma:
# <div class="alert alert-block alert-info">
# <b>Respuesta del estudiante</b> <a class=“tocSkip”></a>
# </div>

# # introducción

# Has recibido una tarea analítica de una tienda en línea internacional. Tus predecesores no consiguieron completarla: lanzaron una prueba A/B y luego abandonaron (para iniciar una granja de sandías en Brasil). Solo dejaron las especificaciones técnicas y los resultados de las pruebas.
# 
# El objetivo de este analisis es evaluar la efectividad de las pruebas A/B realizadas por el equipo anterior, para asi determinar si hubo diferencias significativas en el comportamiento de los usuarios entre los grupos A y B en las pruebas que se mencionan:
# 
# - recommender_system_test
# - interface_eu_test
# 
# Tambien se debe analizar la influencia de campañas de marketing al probar cambios relacionados con la introducción de un sistema de recomendaciones mejorada y evaluar la distribución geográfica, dispositivo utilizado y actividad para detectar posibles sesgos en la asignación de grupos.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Excelente esta introducción! Siempre procura incluir esto para que tu proyectos para que sea más claro el que y cómo de tu proyecto

# In[1]:


# Cargamos todas las librerías
import pandas as pd
import numpy as np
import math as mt 
import matplotlib.pyplot as plt
import seaborn as sns
import statistics as stat
import plotly.express as px 
from scipy import stats as st
from math import factorial
from statsmodels.stats.proportion import proportions_ztest


# In[2]:


#Leemos todos los dataframes y los asignamos en variables
ab_project = pd.read_csv('/datasets/ab_project_marketing_events_us.csv',sep=',')
ab_new_users = pd.read_csv('/datasets/final_ab_new_users_upd_us.csv',sep=',')
ab_events = pd.read_csv('/datasets/final_ab_events_upd_us.csv',sep=',')
ab_participants = pd.read_csv('/datasets/final_ab_participants_upd_us.csv',sep=',')


# # Explorando los datos

# In[3]:


#Hechamos un vistazo rapido a 'ab_project'
ab_project.info()
print()
print(ab_project.head())
print()
print(ab_project.isna().sum())
print()
print(ab_project.describe())
print()
# Encontramos duplicados considerando todas las columnas
print("Valores duplicados: ", ab_project.duplicated(subset=ab_project.columns, keep=False).sum())  # keep=False marca todos los duplicado


# deacuerdo al vistazo rápido de nuestro dataframe project, podemos observar que no tenemos valores ausentes ni duplicados, sin embargo, para cuestiones de análisis en los siguientes pasos, debemos cambiar el tipo de datos para las columnas 'start_dt' y 'finish_dt' ya que se encuentran en tipo Objeto

# In[4]:


#Hechamos un vistazo rapido a 'ab_new_users'
ab_new_users.info()
print()
print(ab_new_users.head())
print()
print(ab_new_users.isna().sum())
print()
print(ab_new_users.describe())
print()
# Encontramos duplicados considerando todas las columnas
print("Valores duplicados: ", ab_new_users.duplicated(subset=ab_new_users.columns, keep=False).sum())  # keep=False marca todos los duplicado


# deacuerdo al vistazo rápido de nuestro dataframe ab_new_users, podemos observar que no tenemos valores ausentes ni duplicados, sin embargo, para cuestiones de análisis en los siguientes pasos, debemos cambiar el tipo de datos para la columna 'first_date' ya que se encuentra en tipo Objeto

# In[5]:


#Hechamos un vistazo rapido a 'ab_events'
ab_events.info()
print()
print(ab_events.head())
print()
print(ab_events.isna().sum())
print()
print(ab_events.describe())
print()
# Encontramos duplicados considerando todas las columnas
print("Valores duplicados: ", ab_events.duplicated(subset=ab_events.columns, keep=False).sum())  # keep=False marca todos los duplicado


# hechando un vistazo rapido al dataframe de ab_events, la columna 'details' tiene 363,447 valores ausentes, sin embargo, notamos que cuando el evento no es purchase, no cuenta con registo ya que no se concretó una compra.
# 
# Hay que cambiar los tipos de datos a fechas para la columna 'event_dt'

# In[6]:


#Hechamos un vistazo rapido a 'ab_participants'
ab_participants.info()
print()
print(ab_participants.head())
print()
print(ab_participants.isna().sum())
print()
print(ab_participants.describe())
print()
# Encontramos duplicados considerando todas las columnas
print("Valores duplicados: ", ab_participants.duplicated(subset=ab_participants.columns, keep=False).sum())  # keep=False marca todos los duplicado


# deacuerdo al vistazo rápido de nuestro dataframe ab_participants, podemos observar que no tenemos valores ausentes ni duplicados
# Sin embargo, el metodo Describe() nos arrojó duplicados en user_id: 14,525 de datos totales vs 13,638 user_id únicos. 
# Probablemente los usuarios participaron en múltiples pruebas con diferente grupo, eso puede contaminar nuestros resultados.
# Vamos a analizar esto mas adelante.

# In[7]:


# Convertimos a datetime
ab_project['start_dt'] = pd.to_datetime(ab_project['start_dt'])
ab_project['finish_dt'] = pd.to_datetime(ab_project['finish_dt'])
ab_new_users['first_date'] = pd.to_datetime(ab_new_users['first_date'])
ab_events['event_dt'] = pd.to_datetime(ab_events['event_dt'])


# In[8]:


# Identificamos duplicados por user_id y group
duplicates_group = ab_participants.groupby(['user_id', 'group']).size().reset_index(name='counts')
duplicates_group = duplicates_group[duplicates_group['counts'] > 1]
print(f"Usuarios duplicados en el mismo grupo: {len(duplicates_group)}")
print(duplicates_group)

# Identificamos duplicados por user_id y ab_test
duplicates_ab = ab_participants.groupby(['user_id', 'ab_test']).size().reset_index(name='counts')
duplicates_ab = duplicates_ab[duplicates_ab['counts'] > 1]
print(f"Usuarios duplicados en la prueba: {len(duplicates_ab)}")


# Una vez analizados los datos que notamos en 'participants', notamos que agrupando por prueba no tenemos duplicados y de los 446 duplicados que se encontrarón al agrupar por grupo son usuarios en diferentes pruebas, esto no afecta los resultados ya que se encuentran en pruebas distintas, no hay necesidad de eliminarlos y 446 refleja menos de un 5% de nuestros datos totales (+14,000 registros)

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Muy buen trabajo con la transformación de datos y la exploración inicial de los mismos, un dataset limpio es clave para una buena toma de decisiones en un experimento.
# </div>

# # Análisis exploratorio de datos

# Primero, necesitamos identificar las etapas del embudo de conversión. estas se encuentran en Events:

# In[9]:


print(ab_events['event_name'].value_counts())


# como podemos ver, el embudo de conversión tendra las siguientes etapas: login, product_page, product_cart, purchase

# In[10]:


# Combinamos dataframes 'ab_events, ab_participants y ab_new_users' para análisis mas sencillo
users_events = ab_events.merge(ab_participants, on='user_id')
users_events = users_events.merge(ab_new_users, on='user_id')

# Identificamos los eventos relevantes en el embudo y los filtramos
funnel_events = ['login', 'product_page', 'product_cart', 'purchase']
funnel_data = ab_events[ab_events['event_name'].isin(funnel_events)]

print(funnel_data)


# In[11]:


# Contamos el número de usuarios en cada etapa del embudo
funnel_conversion = funnel_data.groupby('event_name')['user_id'].nunique().reset_index()
funnel_conversion.columns = ['event_name', 'unique_users']

# Ordenamos los eventos según el orden del embudo
funnel_conversion['event_name'] = pd.Categorical(funnel_conversion['event_name'], categories=funnel_events, ordered=True)
funnel_conversion = funnel_conversion.sort_values('event_name')

# Calculamos la conversión entre etapas Y usamos pct_change para calcular cuánto cambia el número de usuarios entre cada etapa del embudo
funnel_conversion['conversion_rate'] = funnel_conversion['unique_users'].pct_change() * 100 
funnel_conversion['conversion_rate'].fillna(100, inplace=True) 

print(funnel_conversion)


# In[12]:


funnel_conversion = {
    'event_name': ['login', 'product_page', 'product_cart', 'purchase'],
    'unique_users': [58696, 38929, 19284, 19568]
}

# Creamos una grafica de embudo
plt.figure(figsize=(8, 6))
plt.barh(funnel_conversion['event_name'], funnel_conversion['unique_users'], color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
plt.xlabel("Número de usuarios únicos")
plt.ylabel("Etapa del embudo")
plt.title("Gráfica de Embudo de Conversión")
plt.gca().invert_yaxis()  # Invierte el eje y para que el embudo tenga el flujo lógico de arriba a abajo
plt.show()


# Deacuerdo al resultado mostrado, podemos analizar que:
# - **Login:** 58,696 usuarios únicos completaron esta etapa. Al ser la primera etapa, la tasa de conversión es del 100%, osease, todos los usuarios pasan por aqui.
# 
# - **Product_page:** 38,929 usuarios unicos llegaron a esta etapa, la tasa de conversion del -33.68% nos indica que aproximadamente el 33.68% de los usuarios que se registraron no llegaron a ver una página de producto. Quiza tuvieron una mala experiencia con la pagina, por lo cual no pudieron navegar o simplemente perdieron interés.
# 
# - **Product_cart:** 19,284 usuarios únicos añadieron un producto al carrito. La tasa de conversión es -50.46% nos dice que casi la mitad de los usuarios que vieron una página de producto no añadieron un producto al carrito. De nuevo, esto nos dice que quiza tuvieron problemas tecnicos con la pagina o simplemente los productos se les hicieron caros.
# 
# - **Purchase:** 19,568 usuarios únicos realizaron una compra. La tasa de conversión es 1.47% lo cual nos dice que solo un pequeño porcentaje de los usuarios que añadieron un producto al carrito completaron la compra. Quiza algunos usuarios no añadieron el producto al carrito, si no que compraron directamente.
# 

# <div class="alert alert-block alert-warning">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Esta gráfica es excelente para poder mostrar las diferencias entre las etapas del embudo, para mejorar cómo se visualiza la comparación entre grupos también podrías dividir la gráfica con dos colores uno para el grupo A y otro para el B, así desde este punto se empezarían a relizar las comparaciones 
# </div>

# # ¿El número de eventos por usuario está distribuido equitativamente entre las muestras?

# In[13]:


# hacemos un filtro con los datos de eventos de los participantes de la prueba
events_with_group = pd.merge(ab_events, ab_participants, on='user_id', how='inner')

# Contamos el número de eventos por usuario y comparamos la distribucion de eventos
events_per_user = events_with_group.groupby(['user_id', 'group']).size().reset_index(name='event_count')
group_event_dist = events_per_user.groupby('group')['event_count']

print(group_event_dist.describe())

plt.figure(figsize=(10, 6))
sns.boxplot(x='group', y='event_count', data=events_per_user)
plt.title('Distribución de eventos por usuario en cada grupo')
plt.xlabel('Grupo')
plt.ylabel('Número de eventos')
plt.show()


# deacuerdo al analisis que hicimos, el grupo A tiene 7,874 usuarios, mientras que el grupo B tiene 6,205 usuarios. Esto nos dice que el grupo A es ligeramente más grande que el grupo B. Sin embargo, esta diferencia no es significativamente grande, por lo que no debería ser un problema significativo para el análisis.

# In[14]:


# como extra, clasificamos dispositivos en mobile y desktop
desktop = ['PC', 'Mac']
mobile = ['iPhone', 'Android']


users_events['device_type'] = users_events['device'].apply(
    lambda x: 'Mobile' if x in mobile else 'Desktop'
)

# Calculamos tasas de conversión por dispositivo
funnel_device_conv = (
    users_events.groupby(['device_type', 'event_name'])['user_id']
    .nunique()
    .unstack(fill_value=0)
    .T
)

funnel_device_conv = (funnel_device_conv / funnel_device_conv.loc['login']) * 100

print(funnel_device_conv)


# deacuerdo a estos resultados podemos observar que las diferencias de conversiones entre movil y desktop son minimas en cada una de las etapas/eventos

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Correcto, revisar de esta forma ayuda a entender mejor en que plataformas se desempeña mejor o se tiene mejores métricas
# </div>

# # ¿Hay usuarios que están presentes en ambas muestras?

# In[15]:


#creamos un filtro que contiene la diferencia de usuario en cada prueba
users_diff = users_events.groupby(["user_id", "ab_test"])["group"].nunique()
users_diff = users_diff[users_diff > 1].reset_index()

print(f"Usuarios en ambos grupos de la misma prueba: {len(users_diff)}")


# # ¿Cómo se distribuye el número de eventos entre los días?

# In[16]:


# Convertimos event_dt a datetime y extraemos fecha
users_events["event_date"] = pd.to_datetime(users_events["event_dt"]).dt.date

# creamos un filtro de eventos por día y grupo
ab_daily_events = users_events.groupby(["event_date", "group"]).size().reset_index(name="Usuarios")

plt.figure(figsize=(12, 6))
sns.lineplot(x="event_date", y="Usuarios", hue="group", data=ab_daily_events)
plt.title("Eventos de usuarios diarios")
plt.xticks(rotation=45)
plt.show()


# La siguiente gráfica nos muestra la distribución del número de eventos de usuario diarios para los grupos A y B a lo largo del tiempo.Podemos observar que ambos grupos muestran un crecimiento en el número de eventos desde el inicio del periodo hasta alrededor del 21 de diciembre, siendo este dia el pico mas grande de actividad, siendo mas alto en el Grupo A y despues caen considerablemente.
# Esto nos dice que quiza por las fechas, las compras de ultimo momento son mayores este dia por las fechas festivas siguientes, y posterior a esta fecha ya no hay tantas compras.

# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Muy buen trabajo, con esta gráfica se ve correctamente la distribución de los eventos a lo largo del tiempo
# </div>

# # ¿Hay alguna peculiaridad en los datos que hay que tener en cuenta antes de iniciar la prueba A/B?

# In[17]:


# Verificamos la distribución de usuarios entre los grupos
users_per_group = ab_participants.groupby(["ab_test", "group"]).size()
print("\nDistribución de usuarios por grupo:")
print(users_per_group)
    
# Filtramos eventos durante la campaña
eventos_campaña = users_events[(users_events["event_date"] >= pd.to_datetime("2020-12-25")) & (users_events["event_date"] <= pd.to_datetime("2021-01-03"))
]

print(f"Eventos durante toda la campaña: {len(eventos_campaña)} \n({len(eventos_campaña)/len(users_events)*100:.1f}% del total)")


# <div class="alert alert-block alert-danger">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# En esta parte se está intentando utilizar una variable que no ha sido definida, por favor, asegurate que estás usando las variables correctas para garantizar una buena ejecución
# </div>
# 
# <div class="alert alert-block alert-info">
# <b>Corregido, mil disculpas </b> <a class=“tocSkip”></a>
# </div>

# # Evaluar los resultados de la prueba A/B:

# ## ¿Qué puedes decir sobre los resultados de la prueba A/B?
# 

# Deacuerdo a la información previa , podemos observar un gran desequilibrio de datos entre el Grupo A y B de la prueba **recommender_system_test**. Mientras que en **interface_eu_test** muestra una distribución bastante equilibrada entre los grupos A y B. Por lo cual podemos continuar con la prueba A/B con este modelo. Para **recommender_system_test** el desequilibrio puede sesgar los resultados, por lo cual se recomienda reequilibrar los grupos o recolectar más datos para el grupo B antes de analizar los resultados. En el caso de los eventos, solo el 11.2% de los eventos ocurrieron durante el periodo navideño, Esto es muy importante y se debe considerar ya que el comportamiento del usuario puede cambiar significativamente durante las vacaciones.

# ## Utiliza una prueba z para comprobar la diferencia estadística entre las proporciones.

# In[18]:


# hacemos un filtro donde solo tengamos usuarios de la prueba recommender_system_test
events_filt = users_events[users_events['ab_test'] == 'recommender_system_test']

# Función para calcular las conversiones para un evento dado en ambos grupos

def c_conversions(event_name):
    conversion_A = events_filt[(events_filt['group'] == 'A') & (events_filt['event_name'] == event_name)]['user_id'].nunique()
    conversion_B = events_filt[(events_filt['group'] == 'B') & (events_filt['event_name'] == event_name)]['user_id'].nunique()

    n_A = events_filt[events_filt['group'] == 'A']['user_id'].nunique()
    n_B = events_filt[events_filt['group'] == 'B']['user_id'].nunique()

    return [conversion_A, conversion_B], [n_A, n_B]

# prueba z de dos proporciones.
alpha = 0.05  # Nivel de significación
events = ['login', 'product_page', 'product_cart', 'purchase']


for event in events:
    conversions, nobs = c_conversions(event)
    z_stat, p_value = proportions_ztest(conversions, nobs)


    print(f"\nResultados de la prueba z para {event}:")
    print(f"Conversiones Grupo A: {conversions[0]} de {nobs[0]}")
    print(f"Conversiones Grupo B: {conversions[1]} de {nobs[1]}")
    print(f"Estadística z: {z_stat:.4f}")
    print(f"Valor p: {p_value:.4f}")
    print("Significativo" if p_value < alpha else "No significativo")


# <div class="alert alert-block alert-success">
# <b>Comentario del revisor (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# Muy bien! Siempre hay que determinar con pruebas estadísticas si las comparaciones realizadas tienen diferencias significativas para poder tomar las decisiones correctas
# </div>

# como conclusiones de la prueba Z y las conversiones de cada evento:
# 
# - **login**: podemos observar que no hay diferencias significativas entre el grupo A y B en esta etapa, ya que, como se mencionó anteriormente, el 100% de los usuarios pasan por esta etapa.
# 
# - **product_page**: observamos una diferencia estadísticamente significativa entre los grupos A y B. El grupo A tuvo una tasa de conversión mayor que el grupo B.
# 
# - **product_cart**: No se observa una diferencia significativa entre los grupos A y B en este evento.
# 
# - **purchase**: podemos observar una diferencia estadísticamente significativa entre los grupos A y B para el evento purchase. El grupo A tuvo una tasa de conversión mayor que el grupo B.
# 
# **Cabe destacar, como se mencionó anteriormente**, basarnos en los resultados y sacar conclusiones de este analisis no es correcto, es meramente demostrativo, ya que hay demasiada diferencia en nuestros datos para el grupo B en esta prueba.
# Como recomendacion nuevamente, hay que restructurar los datos.

# <div class="alert alert-block alert-danger">
# <b>Comentario general (1ra Iteracion)</b> <a class=“tocSkip”></a>
# 
# En general tienes un muy buen análisis de la prueba AB Rafael, la única razón por la que te regreso el proyecto es porque hubo una celda que está usando una variable que no se definió antes de esa celda lo que causa el error de ejecución y es importante garantizar la correcta ejecución y reproducción de los notebooks en otras instancias. Saludos!
# </div>
# 
# <div class="alert alert-block alert-info">
# <b>Muchas gracias por la revision, se corrigio el error, saludos! </b> <a class=“tocSkip”></a>
# </div>

# <div class="alert alert-block alert-success">
# <b>Comentario general (2da Iteracion)</b> <a class=“tocSkip”></a>
# 
# Muy bien! Ahora se ejecuta todo correctamente Rafael. Siempre hay que asegurar una correcta ejecución de los notebooks ya que a menudo los compartirás con tus compañeros de equipo y no pueden haber celdas que fallen. 
#     
# Por otro lado, una prueba AB es algo que muy a menudo se utiliza y que te vas a encontrar en tu profesión, por lo que es importante que sepas cómo hacerlo y siempre que calcules una métrica mostrarla por cada grupo para poder comparar. Saludos!
# </div>
