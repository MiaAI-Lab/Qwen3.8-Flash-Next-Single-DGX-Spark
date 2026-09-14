#!/usr/bin/env python3
"""Gate de calidad para Flash-Next ANTES de enrutarle trafico.

El hueco que dejo pasar el asturiano: nadie probo espanol. La evidencia de
calidad publicada es MGSM en ingles y chino; ninguna receta publica prueba
de idioma. Esto lo cubre.
"""
import os
import json, re, sys, unicodedata, urllib.request

PORT = os.environ.get("PORT", "8888")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = "qwen3.8-flash-next"

def ask(prompt, mt=1500, temp=0.0):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "temperature": temp, "max_tokens": mt}
    req = urllib.request.Request(URL, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=600).read())
    m = d["choices"][0]["message"]
    u = d["usage"]
    return ((m.get("content") or "").strip(), u["completion_tokens"],
            (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))

# Marcadores asturianos/leoneses que NO aparecen en castellano normal.
ASTUR = [r"\bnun\b", r"\bye\b", r"\bnon\b", r"\bfaer\b", r"\bcomu\b",
         r"\btamien\b", r"\bnamái\b", r"\bguaje\b", r"\bfalar\b", r"\bdrechu\b",
         r"\b\w+u\b(?<!\bsu\b)(?<!\btu\b)(?<!\bmu\b)"]  # -u final masculino

SPANISH_STOP = ["que", "de", "la", "el", "en", "por", "con", "para", "una", "los"]

PRUEBAS = [
    ("es-basico",  "Explica en dos frases qué es la memoria unificada en una GPU integrada."),
    ("es-tecnico", "Describe paso a paso cómo diagnosticarías una fuga de memoria en un servidor Python."),
    ("es-largo",   "Escribe tres párrafos sobre la historia de la informática en España, con nombres y fechas."),
    ("es-acentos", "Escribe una frase que contenga estas palabras: año, corazón, pequeño, mañana, ñu, güisqui."),
    ("es-codigo",  "Escribe una función Python que valide un DNI español, con comentarios en español."),
]

fallos = []
print(f"  {'prueba':<12} {'tok':>5} {'razon':>6}  {'acent':>5} {'roto':>4}  {'astur':>5}  veredicto")
print("  " + "-"*62)
for nombre, p in PRUEBAS:
    try:
        txt, n, rz = ask(p)
    except Exception as e:
        print(f"  {nombre:<12}  ERROR {type(e).__name__}: {e}"); fallos.append(nombre); continue
    if not txt:
        print(f"  {nombre:<12} {n:>5} {rz:>6}  RESPUESTA VACIA"); fallos.append(nombre); continue
    # acentos correctos: hay caracteres no-ASCII y ningun replacement char
    acent = sum(1 for c in txt if ord(c) > 127)
    roto  = txt.count("�")
    # deriva asturiana
    astur = sum(len(re.findall(rx, txt.lower())) for rx in ASTUR[:-1])
    # es castellano de verdad?
    stops = sum(1 for w in SPANISH_STOP if f" {w} " in f" {txt.lower()} ")
    need_acc = nombre == "es-acentos"
    ok = roto == 0 and astur == 0 and stops >= 4 and (acent > 0 or not need_acc)
    if not ok: fallos.append(nombre)
    print(f"  {nombre:<12} {n:>5} {rz:>6}  {acent:>5} {roto:>4}  {astur:>5}  {'OK' if ok else '*** REVISAR ***'}")
    if not ok:
        print(f"      muestra: {txt[:160]!r}")

print()
if fallos:
    print(f"  *** {len(fallos)} prueba(s) a revisar: {', '.join(fallos)}")
    sys.exit(1)
print("  todas las pruebas de espanol pasan")
