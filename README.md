# Taller LSTM: predicción de demanda eléctrica de la hora siguiente

Proyecto para diseñar, comparar y justificar arquitecturas LSTM que predicen la demanda de la hora siguiente a partir de una ventana de *n* horas (12, 24, 48).

## Estructura

```
data/raw/            dataset original (sin modificar)
data/processed/      dataset_limpio.parquet (salida de la Fase 1)
notebooks/           01_limpieza · 02_eda · 03_preparacion · 04_resultados · 05_analisis
src/                 limpieza · preparacion · modelos · metricas · experimentos · graficos · entrenar_todo
resultados/          auditoria_limpieza.csv · baselines.csv · tabla_resultados.csv · modelo_produccion.json
                     corridas/ (métricas e historial por experimento) · modelos/ (.keras) · figuras/
app/app.py           aplicación Streamlit
```

Notebooks, entrenamiento y app usan el mismo código de `src/`, así que el preprocesamiento es idéntico en todos los pasos.

## Cómo reproducir

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt
.venv/Scripts/python -m ipykernel install --user --name taller-lstm
```

1. Ejecute en orden los notebooks `01` → `03`, con el kernel *Python (Taller LSTM)*.
2. Entrene todos los modelos (son 28 corridas y tardan del orden de una hora en CPU). Si se interrumpe, al relanzarlo retoma donde quedó:
   ```bash
   .venv/Scripts/python src/entrenar_todo.py
   ```
3. Ejecute los notebooks `04` y `05`.
4. Abra la aplicación:
   ```bash
   .venv/Scripts/streamlit run app/app.py
   ```

## Resultados principales

| Modelo | Ventana | RMSE test (MW) | R² test |
|---|---|---|---|
| **B. LSTM profunda** (desplegada) | 24 h | **24.13** | **0.906** |
| A. LSTM base | 24 h | 24.54 | 0.903 |
| C. Conv1D + LSTM | 48 h | 24.83 | 0.901 |
| Mejor baseline (misma hora de la semana anterior) | — | 37.24 | 0.777 |
| Persistencia (hora anterior) | — | 44.08 | 0.688 |

- Las LSTM reducen el error en ~35 % respecto del mejor baseline.
- El modelo desplegado se eligió por el menor RMSE de **validación**; el test solo se usó para el reporte final.
- La tabla completa está en `resultados/tabla_resultados.csv`, y las respuestas a las 9 preguntas del taller en `notebooks/05_analisis.ipynb`.

## Reglas metodológicas aplicadas

- Datos ordenados cronológicamente, sin duplicados y con la rejilla horaria completa verificada.
- Ningún valor anómalo se reemplaza por uno estimado. Cada decisión queda registrada en `resultados/auditoria_limpieza.csv`.
- División cronológica: train hasta 2026-06-30, validación de julio a septiembre de 2026 y test de octubre a diciembre de 2026.
- Escaladores ajustados solo con datos de entrenamiento.
- Pruebas automáticas de fuga de información (notebook 03).
- Todas las comparaciones de test se hacen sobre el mismo conjunto de horas, y siempre contra modelos de referencia (baselines).
