# Metodología — U-Net exploratoria para severidad de incendios

## 1. Pregunta científica

¿La **configuración espacial local** del paisaje (LULC previo al incendio) contiene patrones asociados a **baja, media y alta severidad** dentro de cicatrices forestales en Chile?

Este no es un modelo operacional de predicción. Es una herramienta exploratoria para pasar de *dónde quemó* a *qué composición/configuración local del paisaje se asocia con distintos niveles de severidad*.

## 2. Datos de entrada

Por cada incendio `fire_id`:

| Raster | Descripción |
|--------|-------------|
| `lulc.tif` | Cobertura/uso del suelo previo (MapBiomas), categórico |
| `severity.tif` | Severidad post-incendio (dNBR/RdNBR discretizado) |
| `scar.tif` | Máscara binaria de cicatriz |

Estructura esperada:

```
data/raw/fires/{fire_id}/
  lulc.tif
  severity.tif
  scar.tif
```

## 3. Preprocesamiento

1. **Alineación espacial** (`prepare_rasters.py`): reproyección a CRS común (p.ej. UTM), resolución 30 m, extensión común. Resampling *nearest* para variables categóricas.
2. **Codificación LULC**: one-hot encoding — cada clase es un canal independiente (`C × H × W`). No se usan códigos numéricos como valores continuos.
3. **Severidad**: clases 1=baja, 2=media, 3=alta; clase 0 = fuera de cicatriz/sin dato.

## 4. Diseño de ventanas (32 × 32)

- Tamaño: **32 × 32 px** ≈ **960 × 960 m** a 30 m/px.
- **Centro** dentro de la cicatriz con severidad válida.
- **Entrada X**: LULC one-hot, puede incluir contexto fuera de la cicatriz.
- **Salida Y**: severidad observada; solo se evalúa dentro de la cicatriz.
- **Máscara de pérdida**: ignora píxeles fuera de cicatriz o sin severidad.

### Criterios de selección (configurables en `config/default.yaml`)

| Criterio | Parámetro | Default |
|----------|-----------|---------|
| Centro en cicatriz | implícito | — |
| Mínimo % quemado en ventana | `min_burn_fraction` | 0.25 |
| Mínimo % severidad válida | `min_valid_severity_fraction` | 0.20 |
| Máximo % fuera de cicatriz | `max_outside_scar_fraction` | 0.75 |
| Balance baja/media/alta | `balance_severity` | true |
| Muestras por incendio | `samples_per_fire` | 200 |

Cada patch guarda `fire_id`, coordenadas del centro y clase de severidad en el centro.

## 5. Modelo

**SmallUNet** — U-Net compacta para patches 32×32:

- Entrada: `(batch, n_clases_lulc, 32, 32)` one-hot
- Encoder: 2 niveles de pooling → bottleneck 8×8 (configurable)
- Salida: `(batch, 4, 32, 32)` logits (clase 0 ignorada en pérdida)
- Sin normalización continua de LULC — respeta naturaleza categórica

## 6. Función de pérdida

`MaskedCrossEntropyLoss`:

- `ignore_index=0` (fuera de cicatriz)
- Ponderación opcional por frecuencia de clase
- Promedio solo sobre píxeles con `loss_mask=True`

## 7. Validación sin fuga espacial

**Nunca** split aleatorio por píxel/patch sin agrupar.

Estrategias implementadas:

- **`holdout_fires`**: fracción de cicatrices enteras a validación (p.ej. 70/30 por incendio).
- **`leave_one_fire_out`**: entrenar con N−1 cicatrices, validar con 1 completa.
- **`kfold_fires`**: GroupKFold por `fire_id`.

Métricas reportadas por fold: loss, accuracy enmascarada, matrices de confusión.

## 8. Interpretabilidad

Post-entrenamiento (`analyze_results.py`):

1. **Patches de alta confianza** por clase de severidad
2. **Composición LULC** comparada entre clases (proporciones, combustible, riqueza)
3. **Métricas de paisaje** por patch:
   - proporción por cobertura
   - fracción combustible
   - riqueza / Shannon
   - densidad de bordes
   - continuidad de combustibles
   - contacto combustible / no combustible
   - fragmentación del mosaico
4. **Visualizaciones**: tripletes LULC | severidad | predicción
5. **Grad-CAM** sobre capas convolucionales para importancia espacial

## 9. Flujo de trabajo

```text
1. Colocar rasters en data/raw/fires/{fire_id}/
2. python scripts/prepare_rasters.py --fire-id INCENDIO_01
3. python scripts/generate_patches.py
4. python scripts/train.py --fold 0
5. python scripts/analyze_results.py --checkpoint checkpoints/fold_0/best_model.pt
```

## 10. Comunicación para congreso

Productos sugeridos:

- Mapas de ejemplo: patch LULC + severidad observada + predicción
- Barras de composición LULC por nivel de severidad
- Boxplots de métricas de paisaje (borde, continuidad, contacto)
- Grad-CAM en patches de alta severidad
- Tabla de desempeño por cicatriz (leave-one-out)

## 11. Limitaciones explícitas

- Ventanas pequeñas capturan contexto local, no condiciones topográficas o climáticas.
- El desempeño predictivo no es el objetivo principal; interpretar con cautela.
- Clases LULC agrupadas deben definirse según MapBiomas Chile y la escala del análisis.
- Resultados preliminares — validar con más cicatrices y variables adicionales.
