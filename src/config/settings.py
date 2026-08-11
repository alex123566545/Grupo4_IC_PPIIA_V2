
TEST_SIZE = 0.25

RANDOM_STATE = 42

TARGET = "target_riesgo_alto_3sem"


FEATURES = [
    "distrito",
    "altitud_msnm",
    "categoria_zootecnica",
    "raza_predominante",
    "tamano_lote_cabezas",
    "lote_sensorizado",
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
    "media_movil_3s_temperatura",
    "media_movil_3s_pastura",
    "media_movil_3s_condicion_corporal",
    "semana_sin",
    "semana_cos",
]

MODEL_PATH = "models/model.pkl"

ENCODERS_PATH = "models/encoders.pkl"

