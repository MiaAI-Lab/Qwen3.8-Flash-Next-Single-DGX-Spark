#!/usr/bin/env python3
"""Benchmark de Flash-Next con el protocolo estructurado publicado,
para que nuestros numeros sean comparables con otras recetas publicadas.

El protocolo:
  Decode Bench: 400 tokens de completion, temperature=0, top_p=1,
                thinking OFF, servidor ya caliente, por nivel de concurrencia.
  Suite de produccion: 4 clases de carga, 256 tokens, 2 repeticiones,
                temperature=0, thinking off, se reportan medianas.
"""
import os
import json, statistics, sys, time, urllib.request
import concurrent.futures as cf

PORT = os.environ.get("PORT", "8888")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next")

def run(prompt, mt, thinking=False):
    body = {"model": MODEL, "messages": [{"role":"user","content":prompt}],
            "temperature": 0, "top_p": 1, "max_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": thinking}}
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type":"application/json"})
    t0 = time.time()
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    el = time.time() - t0
    return d["usage"]["completion_tokens"], el

COUNT = "Cuenta de uno en uno desde 1 hasta 400, separando los numeros por comas. No digas nada mas."

CLASES = {
 "Agent / tool JSON": "Devuelve solo un JSON valido describiendo tres herramientas de un agente: nombre, descripcion y parametros con tipos.",
 "Code edit":         "Refactoriza esta funcion para que use async/await y maneje errores: def fetch(url): r = requests.get(url); return r.json()",
 "Generic coding":    "Escribe una clase Python LRUCache con get y put en O(1) usando OrderedDict, mas un test.",
 "Long-context review":"Revisa este diseno y senala tres problemas: una API REST sin versionado, sin paginacion, con autenticacion por API key en query string y sin rate limiting.",
}

print("  calentando...", flush=True)
for _ in range(3): run("Di OK", 16)

print()
print("  === Decode Bench (400 tokens, thinking off) ===")
print(f"  {'Concurrencia':<14} {'Por flujo':>12} {'Agregado':>12} {'TTFT aprox':>11}")
print("  " + "-"*54)
for c in (1, 2, 4):
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=c) as ex:
        res = list(ex.map(lambda _: run(COUNT, 400), range(c)))
    wall = time.time() - t0
    toks = sum(r[0] for r in res)
    per  = statistics.median(r[0]/r[1] for r in res)
    print(f"  C{c:<13} {per:>9.2f} t/s {toks/wall:>9.2f} t/s {statistics.median(r[1] for r in res)*1000/max(r[0] for r in res):>8.0f} ms")

print()
print("  === Suite de produccion (256 tokens, 2 reps, medianas) ===")
print(f"  {'Clase':<22} {'C1':>10} {'C4 agregado':>13}")
print("  " + "-"*48)
for nombre, p in CLASES.items():
    c1 = statistics.median([run(p,256)[0]/run(p,256)[1] for _ in range(2)])
    t0 = time.time()
    with cf.ThreadPoolExecutor(max_workers=4) as ex:
        res = list(ex.map(lambda _: run(p,256), range(4)))
    agg = sum(r[0] for r in res)/(time.time()-t0)
    print(f"  {nombre:<22} {c1:>7.1f} t/s {agg:>10.1f} t/s")
