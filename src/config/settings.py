TEST_SIZE = 0.25   # Fase 4 del proyecto SIPREM-BOVINO: hold-out 25% estratificado

RANDOM_STATE = 42

TARGET = "target_riesgo_alto_3sem"   # clasificación binaria de riesgo de mortalidad alta
                                      # en t+1, t+2 o t+3 (ventana de 3 semanas, no solo la
                                      # semana inmediata siguiente -> ver hallazgo de precision baja)

FEATURES = [
    #"id_lote",                       # se usa como grupo en GroupKFold, no como feature
    "distrito",
    "altitud_msnm",
    "categoria_zootecnica",
    "raza_predominante",
    "tamano_lote_cabezas",
    "lote_sensorizado",
    "distancia_centro_veterinario_km",
    "uso_registro_digital",
    "cobertura_vacunacion_pct",
    "dias_desde_desparasitacion",
    "animales_nuevos_30d",
    "visitas_tecnicas_mes",
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
    "actividad_sensor_indice",        # NULL cuando lote_sensorizado=False (ausencia estructural)
    "condicion_corporal_prom",
    "precio_leche_local_s_kg",
    "media_movil_3s_temperatura",
    "media_movil_3s_pastura",
    "media_movil_3s_condicion_corporal",
    #"media_movil_3s_casos_clinicos",               # importancia casi nula, descartada (Etapa 4)
    #"casos_respiratorios_lag1",                     # importancia casi nula, descartada (Etapa 4)
    #"casos_diarreicos_lag1",                        # importancia casi nula, descartada (Etapa 4)
    #"interaccion_desparasitacion_animales_nuevos",  # importancia casi nula, descartada (Etapa 4)
    "semana_sin",
    "semana_cos",
    #"semana_anio",                                # reemplazada por codificación cíclica
    #"muertes_semana_siguiente",                    # fuga: mismo target en otra forma
    #"tasa_mortalidad_semana_siguiente_pct",         # fuga: mismo target en otra forma
    #"riesgo_mortalidad_alta_semana_siguiente",      # fuga: mismo target en otra forma (texto)
    #"target_riesgo_alto",                           # fuga: subconjunto del nuevo target (t+1 solo)
]

MODEL_PATH = "models/model.pkl"

ENCODERS_PATH = "models/encoders.pkl"