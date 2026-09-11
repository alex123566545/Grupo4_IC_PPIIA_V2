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

    "tamano_lote_cabezas",
    "distancia_centro_veterinario_km",

    "cobertura_vacunacion_pct",
    "dias_desde_desparasitacion",
    "casos_respiratorios",
    "casos_diarreicos",

    "temperatura_min_c",
    "temperatura_media_c",
    "temperatura_max_c",
    "humedad_relativa_pct",
    "precipitacion_semanal_mm",

    "condicion_pastura_indice",
    "indice_ndvi_satelital",
    "consumo_ms_kg_animal_dia",
    "agua_l_animal_dia",
    "actividad_sensor_indice",
    "condicion_corporal_prom",
    "precio_leche_local_s_kg",

    "media_movil_4s_temperatura",
    "media_movil_4s_pastura",
    "media_movil_4s_condicion_corporal",

    "semana_sin",
    "semana_cos",

    "animales_nuevos_30d",
    "lote_sensorizado",
    "uso_registro_digital",
]
# ==========================================================
# RUTAS DE LOS ARCHIVOS DEL MODELO
# ==========================================================

MODEL_PATH = "models/model.pkl"
ENCODERS_PATH = "models/encoders.pkl"