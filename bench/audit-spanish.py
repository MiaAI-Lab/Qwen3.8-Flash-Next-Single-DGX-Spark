#!/usr/bin/env python3
"""Auditoria exhaustiva de Flash-Next con vocabulario reducido de 47k.

Disenada contra el fallo concreto que nos llevo a revertirlo en septiembre:
el modelo derivaba al asturiano. Ese tipo de deriva NO se ve en respuestas
cortas, asi que aqui todo es largo, multivuelta y analizado por parrafos.

Detecta:
  - caracteres de reemplazo (byte-fallback roto -> acentos y ñ)
  - marcadores asturianos/leoneses, gallegos, catalanes y portugueses
  - deriva POR PARRAFO, no solo en el conjunto (una respuesta puede empezar
    en castellano y torcerse a la mitad)
  - degradacion a lo largo de una conversacion multivuelta
  - salida vacia por presupuesto de razonamiento
"""
import os
import json, re, sys, time, urllib.request
import functools
print = functools.partial(print, flush=True)

PORT = os.environ.get("PORT", "8888")
URL = f"http://127.0.0.1:{PORT}/v1/chat/completions"
MODEL = os.environ.get("SERVED_MODEL_NAME", "qwen3.8-flash-next")

# Qwen publica ajustes DISTINTOS por modo, y mezclarlos es lo que provoca
# "language mixing" segun su propia advertencia. El generation_config.json del
# checkpoint lleva los de thinking, asi que con thinking OFF hay que pasar los
# de instruct explicitamente o se corre la combinacion desaconsejada.
#   thinking ON : temp 1.0, top_p 0.95, top_k 20, presence_penalty 0.0
#   thinking OFF: temp 0.7, top_p 0.80, top_k 20, presence_penalty 1.5
# https://huggingface.co/Qwen/Qwen3.8-Flash-Next
THINKING = False
SAMPLING = ({"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0}
            if THINKING else
            {"temperature": 0.7, "top_p": 0.80, "top_k": 20, "presence_penalty": 1.5})

def chat(messages, mt=2000, temp=None):
    body = {"model": MODEL, "messages": messages, "max_tokens": mt,
            "chat_template_kwargs": {"enable_thinking": THINKING}, **SAMPLING}
    if temp is not None: body["temperature"] = temp
    req = urllib.request.Request(URL, json.dumps(body).encode(), {"Content-Type": "application/json"})
    d = json.loads(urllib.request.urlopen(req, timeout=900).read())
    m = d["choices"][0]["message"]; u = d["usage"]
    return ((m.get("content") or "").strip(), u["completion_tokens"],
            (u.get("completion_tokens_details") or {}).get("reasoning_tokens", 0))

# --- deteccion de idiomas vecinos -----------------------------------------
MARCADORES = {
 "asturiano": [r"\bnun\b", r"\bye\b(?! )", r"\bfaer\b", r"\bcomu\b", r"\bfalar\b",
               r"\bdrechu\b", r"\bguaje\b", r"\bnamái\b", r"\bmesmu\b", r"\btamién\b(?=\s)",
               r"\b\w{3,}u\b(?<!\bsu\b)(?<!\btu\b)(?<!\bmu\b)(?<!menu\b)(?<!\bflu\b)"],
 "gallego":   [r"\bnon\b", r"\bmáis\b", r"\bconcello\b", r"\bfoi\b", r"\btamén\b", r"\bque\bé\b"],
 "catalan":   [r"\baixò\b", r"\bamb\b", r"\bperò\b", r"\baquest\b", r"\bmés\b"],
 "portugues": [r"\bnão\b", r"\bcomo\bé\b", r"\bvocê\b", r"\bmuito\b", r"\bpalavra\b"],
}
# Ortografia. Un "reescrivi" es castellano impecable, sin caracteres rotos y sin
# marcadores asturianos: pasa todos los filtros de idioma. Se colo por aqui el
# 2026-09-13 y esta lista existe por eso. Solo formas INCORRECTAS.
#
# ESTADO: esta comprobacion NO ha detectado todavia ningun error real. En cuatro
# auditorias solo ha producido falsos positivos de sus propias reglas ('la',
# 'el', 'anduvo', 'hacia'), cada uno corregido al aparecer. El unico error
# confirmado -- "reescrivi" por "reescribi" -- lo encontro un usuario en
# produccion, no este arnes. Tratarla como red de seguridad sin validar, no
# como evidencia de que la ortografia esta bien.
ORTO = [r"\bescriv\w+", r"\breescriv\w+", r"\brecivi\w+", r"\bdeveria\b",
        r"\bestubo\b", r"\bandubo\b", r"\bavia\b(?! )", r"\bhavia\b",
        r"\bubiera\b", r"\bboy\b", r"\bbamos\b", r"\bbolver\w*",
        r"\bprohivi\w+", r"\bconcivi\w+", r"\bexhivi\w+", r"\bmobil\b",
        r"\baser\b"]

STOP_ES = ["que","de","la","el","en","por","con","para","una","los","del","se","es","al"]

def analiza(txt):
    """Devuelve (replacement_chars, {idioma: hits}, parrafos_sospechosos, faltas)."""
    roto = txt.count("�")
    low = txt.lower()
    hits = {k: sum(len(re.findall(rx, low)) for rx in v) for k, v in MARCADORES.items()}
    sosp = []
    def es_prosa(p):
        ls = [l for l in p.strip().split("\n") if l.strip()]
        marcas = sum(1 for l in ls if l.lstrip().startswith(("-","*","|","#","1.","2.","3.")))
        return len(ls) > 0 and marcas / len(ls) < 0.4
    for i, par in enumerate(p for p in txt.split("\n\n") if len(p.strip()) > 120 and es_prosa(p)):
        pl = par.lower()
        stops = sum(1 for w in STOP_ES if f" {w} " in f" {pl} ")
        h = {k: sum(len(re.findall(rx, pl)) for rx in v) for k, v in MARCADORES.items()}
        if stops < 3 or h["asturiano"] > 2 or any(h[k] > 1 for k in ("gallego","catalan","portugues")):
            sosp.append((i, stops, h, par[:110]))
    faltas = [m for rx in ORTO for m in re.findall(rx, low)]
    return roto, hits, sosp, faltas

# --- bateria ---------------------------------------------------------------
LARGOS = [
 ("ensayo-historia", "Escribe un ensayo de al menos mil palabras sobre la historia de la informática en España, desde los primeros ordenadores hasta hoy, con nombres propios, instituciones y fechas concretas."),
 ("ensayo-tecnico",  "Explica en profundidad y con todo detalle cómo funciona la gestión de memoria virtual en Linux: paginación, TLB, fallos de página, reclamación y swap. Extiéndete todo lo que necesites."),
 ("narrativa",       "Escribe un relato largo, de al menos ochocientas palabras, sobre un ingeniero que descubre un fallo crítico en un sistema la noche antes de un lanzamiento. Cuida el lenguaje y la ambientación."),
 ("explicacion",     "Explica a alguien sin formación técnica, con analogías y de forma extensa, qué es un modelo de lenguaje, cómo se entrena y por qué a veces se equivoca."),
 ("formal",          "Redacta un informe técnico formal sobre la migración de una infraestructura de servidores a contenedores, con introducción, análisis de riesgos, plan por fases y conclusiones."),
 ("acentos",         "Escribe tres párrafos usando abundantemente estas palabras y sus variantes: año, corazón, pequeño, mañana, señor, compañía, güisqui, pingüino, cigüeña, güero, España, niño, sueño, diseño."),
 ("codigo-es",       "Escribe un módulo Python completo para gestionar una biblioteca: clases, validación de ISBN, persistencia en JSON y tests. Todos los comentarios y docstrings en español."),
 ("json-es",         "Devuelve únicamente un JSON válido con cinco ciudades españolas, cada una con nombre, provincia, población y una descripción de dos frases en español. Sin texto fuera del JSON."),
]

MULTIVUELTA = [
 "Hola, quiero que me ayudes a diseñar una API REST para una tienda online.",
 "Añade autenticación con JWT y explica cómo gestionarías los refresh tokens.",
 "Ahora explícame cómo harías el versionado de esa API y la migración de clientes antiguos.",
 "Por último, redacta la documentación de tres endpoints en español, con ejemplos.",
 "Resume todo lo que hemos hablado en un párrafo largo.",
]

fallos = []
print(f"  {'prueba':<18} {'tok':>5} {'razon':>6} {'roto':>5} {'astur':>6} {'gal':>4} {'cat':>4} {'pt':>4}  veredicto")
print("  " + "-"*86)

for nombre, p in LARGOS:
    try:
        txt, n, rz = chat([{"role":"user","content":p}])
    except Exception as e:
        print(f"  {nombre:<18} ERROR {type(e).__name__}"); fallos.append(nombre); continue
    if not txt:
        print(f"  {nombre:<18} {n:>5} {rz:>6}  VACIA"); fallos.append(nombre); continue
    roto, h, sosp, faltas = analiza(txt)
    need_acc = nombre == "acentos"
    acc = sum(1 for c in txt if ord(c) > 127)
    ok = roto == 0 and h["asturiano"] <= 2 and all(h[k] <= 1 for k in ("gallego","catalan","portugues")) \
         and not sosp and not faltas and (acc > 0 or not need_acc)
    if not ok: fallos.append(nombre)
    print(f"  {nombre:<18} {n:>5} {rz:>6} {roto:>5} {h['asturiano']:>6} {h['gallego']:>4} {h['catalan']:>4} {h['portugues']:>4}  {'OK' if ok else '*** REVISAR ***'}")
    if faltas: print(f"      ORTOGRAFIA: {faltas}")
    if sosp:
        for i, st, hh, frag in sosp[:2]:
            print(f"      parrafo {i}: stops_es={st} {hh} :: {frag!r}")
    if roto:
        print(f"      *** {roto} caracteres de reemplazo -- byte-fallback roto")

print()
print("  --- conversacion multivuelta (la deriva suele aparecer en turnos tardios) ---")
msgs = []
for t, turno in enumerate(MULTIVUELTA, 1):
    msgs.append({"role":"user","content":turno})
    try:
        txt, n, rz = chat(msgs, mt=1800)
    except Exception as e:
        print(f"  turno {t}: ERROR {type(e).__name__}"); fallos.append(f"turno{t}"); break
    msgs.append({"role":"assistant","content":txt})
    if not txt:
        print(f"  turno {t}: {n:>5} tok, {rz} razon, VACIA"); fallos.append(f"turno{t}"); continue
    roto, h, sosp, faltas = analiza(txt)
    ok = roto == 0 and h["asturiano"] <= 2 and not sosp and not faltas
    if not ok: fallos.append(f"turno{t}")
    print(f"  turno {t:<2} {n:>5} tok  roto={roto} astur={h['asturiano']} gal={h['gallego']}  {'OK' if ok else '*** REVISAR ***'}")
    if faltas: print(f"      ORTOGRAFIA: {faltas}")
    if sosp: print(f"      {sosp[0][3]!r}")

print()
print(f"  contexto acumulado del hilo: {sum(len(m['content']) for m in msgs):,} caracteres")
print()
if fallos:
    print(f"  *** {len(fallos)} a revisar: {', '.join(fallos)}"); sys.exit(1)
print("  === TODAS LAS PRUEBAS PASAN ===")
