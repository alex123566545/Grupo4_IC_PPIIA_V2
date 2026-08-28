# ==========================================================
# SIPREM-BOVINO
# CONFIGURACIÓN DEL MODELO
# ==========================================================

RANDOM_STATE = 42


# ==========================================================
# VARIABLE OBJETIVO
# ==========================================================
#
# El dataset GOLD utiliza una ventana de 4 semanas
# para determinar si ocurre al menos un episodio de
# riesgo de mortalidad alto.
#
# El modelo predice:
#
#     ¿Existirá riesgo alto durante las próximas 4 semanas?
#
# ==========================================================

TARGET = "target_riesgo_alto_4sem"


# ==========================================================
# VARIABLES DE ENTRADA
# ==========================================================
#
# Estas variables representan la información disponible
# en la semana actual para predecir el riesgo futuro.
#
# IMPORTANTE:
#
# El target utiliza una ventana futura de 4 semanas,
# mientras que las medias móviles utilizan información
# histórica disponible hasta la semana actual.
#
# Por eso no existe problema en que las medias móviles
# tengan una ventana de 3 semanas.
#
# ==========================================================

FEATURES = [

    # ------------------------------------------------------
    # CONTEXTO GEOGRÁFICO Y DEL LOTE
    # ------------------------------------------------------

    #"distrito",

    #"altitud_msnm",

    #"categoria_zootecnica",

    #"raza_predominante",

    "tamano_lote_cabezas",

    "distancia_centro_veterinario_km",


    # ------------------------------------------------------
    # MANEJO SANITARIO
    # ------------------------------------------------------

    "cobertura_vacunacion_pct",

    "dias_desde_desparasitacion",

    "casos_respiratorios",

    "casos_diarreicos",


    # ------------------------------------------------------
    # VARIABLES CLIMÁTICAS
    # ------------------------------------------------------

    "temperatura_min_c",

    "temperatura_media_c",

    "temperatura_max_c",

    "humedad_relativa_pct",

    "precipitacion_semanal_mm",


    # ------------------------------------------------------
    # ALIMENTACIÓN Y CONDICIÓN DEL LOTE
    # ------------------------------------------------------

    "condicion_pastura_indice",

    "indice_ndvi_satelital",

    "consumo_ms_kg_animal_dia",

    "agua_l_animal_dia",

    "actividad_sensor_indice",

    "condicion_corporal_prom",

    "precio_leche_local_s_kg",


    # ------------------------------------------------------
    # TENDENCIAS HISTÓRICAS
    # ------------------------------------------------------
    #
    # Estas variables resumen las últimas 3 semanas,
    # incluyendo la semana actual.
    #
    # No miran hacia el futuro.
    #
    # El "3sem" aquí NO significa que el target sea de
    # 3 semanas.
    #
    # ------------------------------------------------------

    "media_movil_4s_temperatura",

    "media_movil_4s_pastura",

    "media_movil_4s_condicion_corporal",


    # ------------------------------------------------------
    # ESTACIONALIDAD
    # ------------------------------------------------------

    "semana_sin",

    "semana_cos",


    # ------------------------------------------------------
    # VARIABLES DE MENOR IMPORTANCIA
    # ------------------------------------------------------
    #
    # Se mantienen porque forman parte del conjunto de
    # variables evaluado en el análisis de importancia.
    #
    # Importancia observada:
    #
    # animales_nuevos_30d       ≈ 0.0052
    # lote_sensorizado          ≈ 0.0051
    # uso_registro_digital      ≈ 0.0050
    #
    # ------------------------------------------------------

    "animales_nuevos_30d",

    "lote_sensorizado",

    "uso_registro_digital",
]


# ==========================================================
# RUTAS DE LOS ARCHIVOS DEL MODELO
# ==========================================================

MODEL_PATH = "models/model.pkl"

ENCODERS_PATH = "models/encoders.pkl"