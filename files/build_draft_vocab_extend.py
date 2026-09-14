#!/usr/bin/env python3
"""Extiende el vocabulario de drafting que envia upstream con cobertura espanola.

Se construyo despues de que un intento anterior (reconstruir desde cero con un
diccionario espanol como corpus) tirara 36.160 ids, entre ellos 90 tokens de
byte-fallback -- los que BPE usa para armar la ñ y los acentos.

Tres reglas que salen de aquel fallo:
  1. El 47k de upstream entra ENTERO como suelo. Nunca se pierde lo que funciona.
  2. El rango de byte-fallback se fija incondicionalmente, pase lo que pase con
     las frecuencias.
  3. Solo se anaden ids por FRECUENCIA sobre texto real. Nunca un diccionario:
     alli cada forma flexionada pesa igual que "que".
"""
import argparse, json, pathlib, sys
from collections import Counter

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True, help="vocabulario de upstream (suelo)")
ap.add_argument("--corpus", nargs="+", required=True, help="ficheros de texto; sufijo :N para repetir")
ap.add_argument("--size", type=int, default=65536)
ap.add_argument("--model", default="Mia-AiLab/Qwen3.8-Flash-Next-NVFP4")
ap.add_argument("--byte-fallback-max", type=int, default=400,
                help="todo id por debajo de esto se conserva siempre")
ap.add_argument("--out", required=True)
a = ap.parse_args()

from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=True)

base = {int(l) for l in pathlib.Path(a.base).read_text().split() if l.strip()}
print(f"  suelo (upstream): {len(base):,} ids")

cnt = Counter()
CH = 1 << 20
for spec in a.corpus:
    path, rep = (spec.rsplit(":", 1) + ["1"])[:2] if ":" in spec and spec.rsplit(":",1)[1].isdigit() else (spec, "1")
    rep = int(rep)
    p = pathlib.Path(path)
    for _ in range(rep):
        if p.suffix == ".jsonl":
            for line in p.open(encoding="utf-8"):
                t = json.loads(line).get("text", "")
                if t: cnt.update(tok(t, add_special_tokens=False)["input_ids"])
        else:
            with p.open(encoding="utf-8") as f:
                while True:
                    chunk = f.read(CH)
                    if not chunk: break
                    cnt.update(tok(chunk, add_special_tokens=False)["input_ids"])
    print(f"  {p.name} x{rep}: acumulado {sum(cnt.values()):,} ocurrencias, {len(cnt):,} ids distintos")

# byte-fallback y especiales, incondicionalmente
pinned = {i for i in range(a.byte_fallback_max)}
pinned |= set(tok.all_special_ids or [])
print(f"  fijados incondicionalmente: {len(pinned):,} (byte-fallback + especiales)")

vocab = set(base) | pinned
if len(vocab) > a.size:
    sys.exit(f"  el suelo + fijados ({len(vocab):,}) ya supera --size {a.size:,}")

hueco = a.size - len(vocab)
extra = [i for i, _ in cnt.most_common() if i not in vocab][:hueco]
vocab |= set(extra)
print(f"  anadidos por frecuencia: {len(extra):,}  -> total {len(vocab):,}")

tot = sum(cnt.values())
cub = sum(c for i, c in cnt.items() if i in vocab)
print(f"  cobertura sobre el corpus: {cub/tot*100:.3f}%")
print(f"  byte-fallback conservados (<{a.byte_fallback_max}): {len([i for i in vocab if i < a.byte_fallback_max])}")

o = pathlib.Path(a.out); o.parent.mkdir(parents=True, exist_ok=True)
o.write_text("\n".join(str(i) for i in sorted(vocab)) + "\n")
print(f"  escrito {len(vocab):,} ids -> {o}")
