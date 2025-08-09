
# docfix-poc v3 (Robust auto-params)

Focus:
- Mejor manejo de **iluminación desigual** y **bordes con bajo contraste**.
- Auto-ajuste de parámetros de Hough y Canny (multi-try).
- Supresión de grid más adaptativa.
- Refinamiento en 2 etapas del radio (búsqueda amplia → estrecha).
- Fallback por contornos si Hough falla.
- (Opcional) Procesamiento por lote: si `image` es una carpeta, procesa todas las imágenes soportadas.

## Instalar
```bash
pip install -r requirements.txt
```

## Usar (imagen única)
```bash
python main.py ruta/imagen.jpg --outdir out
```

## Usar (carpeta)
```bash
python main.py ruta/carpeta --outdir out
```

### Flags útiles
- `--minr / --maxr` → forzar rango de radios si lo conoces.
- `--scale 0.8` → acelerar en imágenes muy grandes.
- `--save-steps` → guarda pasos intermedios en `out/debug_*` (útil para tuning).

### Salidas
- `rectified.png`
- `overlay.png`
- `result.json` con:
  - `center [x,y]` (px)
  - `radius` (px)
  - `confidence` (0–1)
  - `method` usado (`hough_refined` | `hough_coarse` | `contour_fallback` | `none`)
