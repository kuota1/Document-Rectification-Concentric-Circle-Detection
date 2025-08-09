import argparse, json, sys
from .pipeline import process_path

def cli(argv=None):
    ap = argparse.ArgumentParser(prog="docfix",
        description="Rectify document photos and detect concentric circles (grid+circle).")
    ap.add_argument("path", help="Imagen o carpeta")
    ap.add_argument("--outdir", default="out", help="Carpeta de salida")
    ap.add_argument("--minr", type=int, default=None, help="Radio mínimo esperado (px)")
    ap.add_argument("--maxr", type=int, default=None, help="Radio máximo esperado (px)")
    ap.add_argument("--scale", type=float, default=1.0, help="Escala previa (1.0 = sin cambio)")
    ap.add_argument("--save-steps", action="store_true", help="Guardar pasos intermedios (debug)")
    args = ap.parse_args(argv)

    outputs = process_path(args.path, args.outdir,
                           minr=args.minr, maxr=args.maxr,
                           scale=args.scale, save_steps=args.save_steps)
    print(json.dumps({"processed": len(outputs), "outdir": args.outdir}, indent=2))

if __name__ == "__main__":
    cli()
