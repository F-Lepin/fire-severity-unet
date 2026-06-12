# fire-severity-unet

Análisis exploratorio de **severidad de incendios forestales** en Chile usando una U-Net compacta sobre patches locales de LULC (MapBiomas).

El objetivo no es predicción operacional, sino identificar **patrones espaciales locales** de cobertura previa al incendio asociados a baja, media y alta severidad dentro de cicatrices.

Documentación metodológica completa: [docs/METHODOLOGY.md](docs/METHODOLOGY.md).

## Instalación

```bash
cd fire-severity-unet
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -e .
```

## Estructura de datos

```
data/raw/fires/{fire_id}/
  lulc.tif       # MapBiomas LULC previo al incendio
  severity.tif   # Severidad discretizada (1=baja, 2=media, 3=alta)
  scar.tif       # Máscara de cicatriz (>0 = quemado)
```

Ajusta clases LULC, CRS y criterios de muestreo en `config/default.yaml`.

## Pipeline

```bash
# 1. Alinear rasters por incendio
python scripts/prepare_rasters.py --fire-id MI_INCENDIO

# 2. Generar patches 32×32 (one-hot LULC + máscara de pérdida)
python scripts/generate_patches.py

# 3. Entrenar con validación por cicatriz (leave-one-fire-out)
python scripts/train.py --fold 0

# 4. Interpretabilidad y figuras
python scripts/analyze_results.py --checkpoint checkpoints/fold_0/best_model.pt
```

## Diseño técnico resumido

| Componente | Detalle |
|------------|---------|
| Patch | 32×32 px (~960 m), centrado en cicatriz |
| Entrada X | LULC one-hot (`8 × 32 × 32` por defecto) |
| Salida Y | Severidad multiclase; clase 0 ignorada |
| Modelo | SmallUNet, 2 niveles de pooling |
| Validación | Por incendio completo (sin fuga espacial) |
| Interpretación | Métricas de paisaje, top patches, Grad-CAM |

## Layout del repositorio

```
config/           Configuración YAML
docs/             Metodología
scripts/          CLI: preparar, patches, entrenar, analizar
src/fire_severity/
  data/           Alineación, encoding, patches, dataset
  models/         SmallUNet
  training/       Pérdida enmascarada, trainer
  validation/     Splits por cicatriz
  interpretability/  Métricas de paisaje, Grad-CAM, figuras
```

## Relación con MapBiomas Fire

Repositorio independiente de [mapbiomas-chile/fire](https://github.com/mapbiomas-chile/fire). FIRE mapea áreas quemadas; este proyecto explora **severidad condicionada al paisaje previo** dentro de cicatrices ya delimitadas.
