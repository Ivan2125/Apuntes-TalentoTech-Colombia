## FUNCIONES DE UTILIDAD PARA EL ETL Y EDA
# Importaciones
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def verifica_duplicados_por_columna(df, columna):
    """
    Verifica y muestra filas duplicadas en un DataFrame basado en una columna específica.

    Esta función toma como entrada un DataFrame y el nombre de una columna específica.
    Luego, identifica las filas duplicadas basadas en el contenido de la columna especificada,
    las filtra y las ordena para una comparación más sencilla.

    Parameters:
        df (pandas.DataFrame): El DataFrame en el que se buscarán filas duplicadas.
        columna (str): El nombre de la columna basada en la cual se verificarán las duplicaciones.

    Returns:
        pandas.DataFrame or str: Un DataFrame que contiene las filas duplicadas filtradas y ordenadas,
        listas para su inspección y comparación, o el mensaje "No hay duplicados" si no se encuentran duplicados.
    """
    # Se filtran las filas duplicadas
    duplicated_rows = df[df.duplicated(subset=columna, keep=False)]
    if duplicated_rows.empty:
        return "No hay duplicados"

    # se ordenan las filas duplicadas para comparar entre sí
    duplicated_rows_sorted = duplicated_rows.sort_values(by=columna)
    return duplicated_rows_sorted


def verificar_tipo_variable(df):
    """
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
    """

    mi_dict = {"nombre_campo": [], "tipo_datos": []}

    for columna in df.columns:
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
    df_info = pd.DataFrame(mi_dict)

    return df_info


def convertir_a_time(x):
    """
    Convierte un valor a un objeto de tiempo (time) de Python si es posible.

    Esta función acepta diferentes tipos de entrada y trata de convertirlos en objetos de tiempo (time) de Python.
    Si la conversión no es posible, devuelve None.

    Parameters:
        x (str, datetime, or any): El valor que se desea convertir a un objeto de tiempo (time).

    Returns:
        datetime.time or None: Un objeto de tiempo (time) de Python si la conversión es exitosa,
        o None si no es posible realizar la conversión.
    """
    if isinstance(x, str):
        try:
            return datetime.strptime(x, "%H:%M:%S").time()
        except ValueError:
            return None
    elif isinstance(x, datetime):
        return x.time()
    return x


def imputa_valor_frecuente(df, columna):
    """
    Imputa los valores faltantes en una columna de un DataFrame con el valor más frecuente.

    Esta función reemplaza los valores "SD" con NaN en la columna especificada,
    luego calcula el valor más frecuente en esa columna y utiliza ese valor
    para imputar los valores faltantes (NaN).

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna a ser imputada.
        columna (str): El nombre de la columna en la que se realizará la imputación.

    Returns:
        None
    """
    # Se reemplaza "SD" con NaN en la columna
    df[columna] = df[columna].replace("SD", pd.NA)

    # Se calcula el valor más frecuente en la columna
    valor_mas_frecuente = df[columna].mode().iloc[0]
    print(f"El valor mas frecuente es: {valor_mas_frecuente}")

    # Se imputan los valores NaN con el valor más frecuente
    df[columna].fillna(valor_mas_frecuente, inplace=True)


def imputa_edad_media_segun_sexo(df, col, agr):
    """
    Imputa valores faltantes en la columna 'edad' utilizando la edad promedio según el género.

    Esta función reemplaza los valores "SD" con NaN en la columna 'edad', calcula la edad promedio
    para cada grupo de género (Femenino y Masculino), imprime los promedios calculados y
    luego llena los valores faltantes en la columna 'edad' utilizando el promedio correspondiente
    al género al que pertenece cada fila en el DataFrame.

    Parameters:
        df (pandas.DataFrame): El DataFrame que contiene la columna 'edad' a ser imputada.

    Returns:
        None
    """

    # Se reemplaza "SD" con NaN en la columna 'edad'
    df[col] = df[col].replace("SD", pd.NA)

    # Se calcula el promedio de edad para cada grupo de género
    promedio_por_genero = df.groupby(agr)[col].mean()
    print(
        f'La edad promedio de Femenino es {round(promedio_por_genero["FEMENINO"])} y de Masculino es {round(promedio_por_genero["MASCULINO"])}'
    )

    # Se llenan los valores NaN en la columna 'edad' utilizando el promedio correspondiente al género
    df[col] = df.apply(
        lambda row: (promedio_por_genero[row[agr]] if pd.isna(row[col]) else row[col]),
        axis=1,
    )
    # Lo convierte a entero
    df[col] = df[col].astype(int)


def verificar_tipo_datos_y_nulos(df):
    """
    Realiza un análisis de los tipos de datos y la presencia de valores nulos en un DataFrame.

    Esta función toma un DataFrame como entrada y devuelve un resumen que incluye información sobre
    los tipos de datos en cada columna, el porcentaje de valores no nulos y nulos, así como la
    cantidad de valores nulos por columna.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        pandas.DataFrame: Un DataFrame que contiene el resumen de cada columna, incluyendo:
        - 'nombre_campo': Nombre de cada columna.
        - 'tipo_datos': Tipos de datos únicos presentes en cada columna.
        - 'no_nulos_%': Porcentaje de valores no nulos en cada columna.
        - 'nulos_%': Porcentaje de valores nulos en cada columna.
        - 'nulos': Cantidad de valores nulos en cada columna.
    """

    mi_dict = {
        "nombre_campo": [],
        "tipo_datos": [],
        "no_nulos_%": [],
        "nulos_%": [],
        "nulos": [],
    }

    for columna in df.columns:
        porcentaje_no_nulos = (df[columna].count() / len(df)) * 100
        mi_dict["nombre_campo"].append(columna)
        mi_dict["tipo_datos"].append(df[columna].apply(type).unique())
        mi_dict["no_nulos_%"].append(round(porcentaje_no_nulos, 2))
        mi_dict["nulos_%"].append(round(100 - porcentaje_no_nulos, 2))
        mi_dict["nulos"].append(df[columna].isnull().sum())
        mi_dict["NaN"].append(df[columna].isnan().sum())

    df_info = pd.DataFrame(mi_dict)

    return df_info.sort_values(ascending=False, by="nulos_%")


def distribucion_edad(df):
    """
    Genera un gráfico con un histograma y un boxplot que muestran la distribución de la edad de los involucrados en los accidentes.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico con un histograma y un boxplot.
    """
    # Se crea una figura con un solo eje x compartido
    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)

    # Se grafica el histograma de la edad
    sns.histplot(df["edad"], kde=True, ax=ax[0])
    ax[0].set_title("Histograma de edad")
    ax[0].set_ylabel("Frecuencia")

    # Se grafica el boxplot de la edad
    sns.boxplot(x=df["edad"], ax=ax[1])
    ax[1].set_title("Boxplot de edad")
    ax[1].set_xlabel("edad")

    # Se ajusta y muestra el gráfico
    plt.tight_layout()
    plt.show()


def cantidades_accidentes_por_anio_y_sexo(df):
    """
    Genera un gráfico de barras que muestra la cantidad_accidentes por año y sexo.

    Parameters:
        df: El conjunto de datos de accidentes.

    Returns:
        Un gráfico de barras.
    """
    # Se crea el gráfico de barras
    plt.figure(figsize=(12, 4))
    sns.barplot(
        x="anio",
        y="edad",
        hue="sexo",
        data=df,
    )

    plt.title("Cantidad de accidentes por año y sexo")
    plt.xlabel("Año")
    plt.ylabel("Edad de las víctimas")
    plt.legend(title="Sexo")

    # Se muestra el gráfico
    plt.show()


def edad_y_rol_victimas(df):
    """
    Genera un gráfico de la distribución de la edad de las víctimas por rol.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    plt.figure(figsize=(8, 4))
    sns.boxplot(y="Rol", x="edad", data=df)
    plt.title("edades por Condición")
    plt.show()


def distribucion_edad_por_victima(df):
    """
    Genera un gráfico de la distribución de la edad de las víctimas por tipo de vehículo.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se crea el gráfico de boxplot
    plt.figure(figsize=(14, 6))
    sns.boxplot(x="Víctima", y="edad", data=df)

    plt.title("Boxplot de edades de Víctimas por tipo de vehículo que usaba")
    plt.xlabel("Tipo de vehiculo")
    plt.ylabel("edad de las Víctimas")

    plt.show()


def cantidad_accidentes_sexo(df):
    """
    Genera un resumen de la cantidad_accidentes por sexo de los conductores.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad_accidentes por sexo de los conductores en orden descendente.
    * Un DataFrame que muestra la cantidad y el porcentaje de accidentes por sexo de los conductores.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se convierte la columna 'fecha' a tipo de dato datetime
    df["fecha"] = pd.to_datetime(df["fecha"])

    # Se extrae el día de la semana (0 = lunes, 6 = domingo)
    df["dia_semana"] = df["fecha"].dt.dayofweek

    # Se crea una columna 'tipo_dia' para diferenciar entre semana y fin_semana
    df["tipo_dia"] = df["dia_semana"].apply(
        lambda x: "fin_semana" if x >= 5 else "Semana"
    )

    # Se cuenta la cantidad_accidentes por tipo_dia
    data = df["tipo_dia"].value_counts().reset_index()
    data.columns = ["tipo_dia", "cantidad_accidentes"]

    # Se crea el gráfico de barras
    plt.figure(figsize=(6, 4))
    ax = sns.barplot(x="tipo_dia", y="cantidad_accidentes", data=data)

    ax.set_title("cantidad_accidentes por tipo_dia")
    ax.set_xlabel("tipo_dia")
    ax.set_ylabel("cantidad_accidentes")

    # Se agrega las cantidades en las barras
    for index, row in data.iterrows():
        ax.annotate(
            f'{row["cantidad_accidentes"]}',
            (index, row["cantidad_accidentes"]),
            ha="center",
            va="bottom",
        )

    # Se muestra el gráfico
    plt.show()


def accidentes_tipo_de_calle(df):
    """
    Genera un resumen de los accidentes de tráfico por tipo de calle y cruce.

    Esta función toma un DataFrame como entrada y genera un resumen que incluye:

    * Un gráfico de barras que muestra la cantidad de víctimas por tipo de calle.
    * Un gráfico de barras que muestra la cantidad de víctimas en cruces.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas por tipo de calle.
    * Un DataFrame que muestra la cantidad y el porcentaje de víctimas en cruces.

    Parameters:
        df (pandas.DataFrame): El DataFrame que se va a analizar.

    Returns:
        None
    """
    # Se crea el gráfico
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    sns.countplot(data=df, x="tipo_calle", ax=axes[0], hue="tipo_calle")
    axes[0].set_title("Cantidad de víctimas por tipo de calle")
    axes[0].set_ylabel("Cantidad de víctimas")

    sns.countplot(data=df, x="cruce", ax=axes[1], hue="tipo_calle")
    axes[1].set_title("Cantidad de víctimas en cruces")
    axes[1].set_ylabel("Cantidad de víctimas")

    # Mostramos los gráficos
    plt.show()

    # # Se calcula la cantidad de víctimas por tipo de calle
    # tipo_calle_counts = df['Tipo de calle'].value_counts().reset_index()
    # tipo_calle_counts.columns = ['Tipo de calle', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por tipo de calle
    # tipo_calle_counts['Porcentaje de víctimas'] = round((tipo_calle_counts['Cantidad de víctimas'] / tipo_calle_counts['Cantidad de víctimas'].sum()) * 100,2)

    # # Se calcula la cantidad de víctimas por cruce
    # cruce_counts = df['Cruce'].value_counts().reset_index()
    # cruce_counts.columns = ['Cruce', 'Cantidad de víctimas']

    # # Se calcula el porcentaje de víctimas por cruce
    # cruce_counts['Porcentaje de víctimas'] = round((cruce_counts['Cantidad de víctimas'] / cruce_counts['Cantidad de víctimas'].sum()) * 100,2)

    # # Se crean DataFrames para tipo de calle y cruce
    # df_tipo_calle = pd.DataFrame(tipo_calle_counts)
    # df_cruce = pd.DataFrame(cruce_counts)

    # #  Se muestran los DataFrames resultantes
    # print("Resumen por Tipo de Calle:")
    # print(df_tipo_calle)
    # print("\nResumen por Cruce:")
    # print(df_cruce)


def graficos_eda_categoricos(cat):
    """
    Realiza gráficos de barras horizontales para explorar datos categóricos.

    Parámetros:
    - cat (DataFrame): DataFrame que contiene variables categóricas a visualizar.

    Retorna:
    - None: La función solo genera gráficos y no devuelve valores.

    La función toma un DataFrame con variables categóricas y genera gráficos de barras horizontales
    para visualizar la distribución de categorías en cada variable. Los gráficos se organizan en
    filas y columnas para facilitar la visualización.
    """
    # Calculamos el número de filas que necesitamos
    from math import ceil

    filas = ceil(cat.shape[1] / 2)

    # Definimos el gráfico
    f, ax = plt.subplots(nrows=filas, ncols=2, figsize=(16, filas * 6))

    # Aplanamos para iterar por el gráfico como si fuera de 1 dimensión en lugar de 2
    ax = ax.flat

    # Creamos el bucle que va añadiendo gráficos
    for cada, variable in enumerate(cat):
        cat[variable].value_counts().plot.barh(ax=ax[cada])
        ax[cada].set_title(variable, fontsize=12, fontweight="bold")
        ax[cada].tick_params(labelsize=12)


def estadisticos_cont(num):
    """
    Calcula estadísticas descriptivas para variables numéricas.

    Parámetros:
    - num (DataFrame o Series): Datos numéricos para los cuales se desean calcular estadísticas.

    Retorna:
    - DataFrame: Un DataFrame que contiene estadísticas descriptivas, incluyendo la media, la desviación estándar,
      los percentiles, el mínimo, el máximo y la mediana.

    La función toma datos numéricos y calcula estadísticas descriptivas, incluyendo la media, desviación estándar,
    percentiles (25%, 50%, 75%), mínimo, máximo y mediana. Los resultados se presentan en un DataFrame organizado
    para una fácil interpretación.

    Nota:
    - El DataFrame de entrada debe contener solo variables numéricas para obtener resultados significativos.
    """
    # Calculamos describe
    estadisticos = num.describe().T
    # Añadimos la mediana
    estadisticos["median"] = num.median()
    # Reordenamos para que la mediana esté al lado de la media
    estadisticos = estadisticos.iloc[:, [0, 1, 8, 2, 3, 4, 5, 6, 7]]
    # Lo devolvemos
    return estadisticos
