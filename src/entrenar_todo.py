"""Lanza todos los experimentos del taller. Uso:  python src/entrenar_todo.py"""
import os
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pandas as pd
import preparacion as P
import experimentos as E

E.DIR_RES.mkdir(parents=True, exist_ok=True)
E.DIR_MOD.mkdir(parents=True, exist_ok=True)
df = pd.read_parquet(E.RAIZ / "data/processed/dataset_limpio.parquet")
comun = E.conjunto_comun_test(df)

corridas = []
# 1) Grilla principal: 3 arquitecturas x 3 ventanas x 2 tratamientos (18 corridas).
for trat in ("B", "A"):
    for arq in ("A", "C", "B"):
        for n in P.VENTANAS:
            corridas.append(dict(id_corrida=f"grid_{arq}_n{n}_{trat}", arq=arq, n=n, tratamiento=trat))
# 2) Pregunta 6: ¿más neuronas = mejor? (A, n=24)
for u in (16, 32, 128, 256):
    corridas.append(dict(id_corrida=f"unidades_A_{u}", arq="A", n=24, kwargs_modelo={"unidades": u}, guardar_modelo=False))
# 3) Pregunta 4: Dropout (B sin dropout, n=24)
corridas.append(dict(id_corrida="dropout_B_sin", arq="B", n=24, kwargs_modelo={"dropout": 0.0}, guardar_modelo=False))
# 4) Pregunta 5: sin Early Stopping (60 épocas fijas: ~3 veces lo que necesita con Early Stopping)
corridas.append(dict(id_corrida="sin_es_B_sin_dropout", arq="B", n=24, early_stopping=False,
                     kwargs_modelo={"dropout": 0.0}, guardar_modelo=False, max_epocas=60))
corridas.append(dict(id_corrida="sin_es_A", arq="A", n=24, early_stopping=False, guardar_modelo=False, max_epocas=60))
# 5) Pregunta 8: ablación de variables (A, n=24)
corridas.append(dict(id_corrida="ablacion_sin_clima", arq="A", n=24, exogenas=["precio_kwh"], guardar_modelo=False))
corridas.append(dict(id_corrida="ablacion_sin_precio", arq="A", n=24,
                     exogenas=["temperatura_c", "humedad_pct", "radiacion_wm2"], guardar_modelo=False))
corridas.append(dict(id_corrida="ablacion_con_viento_lluvia", arq="A", n=24,
                     exogenas=P.EXOGENAS + ["viento_kmh", "precipitacion_mm"], guardar_modelo=False))

# Paralelismo opcional:  python src/entrenar_todo.py <parte> <total>  ejecuta solo las corridas
# cuyo índice % total == parte (p. ej. 3 procesos: 0 3, 1 3, 2 3).
if len(sys.argv) == 3:
    parte, total = int(sys.argv[1]), int(sys.argv[2])
    corridas = [c for k, c in enumerate(corridas) if k % total == parte]
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(max(1, (os.cpu_count() or 4) // total))

t_total = time.time()
for i, c in enumerate(corridas, 1):
    ya = (E.DIR_RES / f"{c['id_corrida']}.json").exists()
    t0 = time.time()
    r = E.entrenar(df, comun=comun, **c)
    m = r["metricas_test_comun"]
    print(f"[{i}/{len(corridas)}] {c['id_corrida']:28s} {'(ya existía)' if ya else f'{time.time()-t0:6.0f}s'}"
          f"  épocas={r['epocas_entrenadas']:3d} mejor={r['mejor_epoca']:3d}"
          f"  RMSE_test={m['RMSE']:.2f}  R2={m['R2']:.3f}", flush=True)
print(f"FIN total {time.time()-t_total:.0f}s", flush=True)
