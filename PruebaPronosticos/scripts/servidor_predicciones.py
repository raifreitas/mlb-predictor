# -*- coding: utf-8 -*-
"""Mini interfaz local de pronosticos MLB.

Sirve en http://localhost:8000 una pagina que muestra las
predicciones de hoy (dbo.Predicciones) con su estado actual
(PENDIENTE / GANADA / PERDIDA / PUSH) y un resumen de los
ultimos 15 dias. La pagina se auto-refresca cada 30 segundos
leyendo /api (consulta directa a SQL Server, sin archivos).

Uso:
    python scripts\\servidor_predicciones.py [puerto]
"""

import json
import sys
import time
import urllib.request
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pyodbc

CONNECTION_STRING_TEMPLATE = (
    "DRIVER={{{driver}}};"
    "SERVER=RAI-FREITAS\\SQLEXPRESS;"
    "DATABASE=MLB_Predictive;"
    "Trusted_Connection=yes;"
    "TrustServerCertificate=yes;"
)

DRIVERS_PREFERIDOS = [
    "ODBC Driver 18 for SQL Server",
    "ODBC Driver 17 for SQL Server",
    "ODBC Driver 13 for SQL Server",
    "SQL Server Native Client 11.0",
]

PUERTO = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
CUOTA_PAGO_POR_UNIDAD = 100.0 / 110.0  # -110 americano (supuesto de pago)


def obtener_driver_odbc():
    disponibles = pyodbc.drivers()
    for preferido in DRIVERS_PREFERIDOS:
        if preferido in disponibles:
            return preferido
    if disponibles:
        return disponibles[0]
    raise RuntimeError("No se encontro un driver ODBC de SQL Server instalado.")


def conectar():
    return pyodbc.connect(CONNECTION_STRING_TEMPLATE.format(
        driver=obtener_driver_odbc()))


def pl_jugada(estado, unidades, cuota):
    """P/L de una jugada. Con cuota decimal guardada se calcula exacto;
    si no hay cuota, se asume pago -110 (fallback historico)."""
    if estado == "GANADA":
        if cuota:
            return unidades * (cuota - 1.0)
        return unidades * CUOTA_PAGO_POR_UNIDAD
    if estado == "PERDIDA":
        return -unidades
    return 0.0  # PUSH o PENDIENTE


_ULTIMA_CARGA_VIVO = 0.0
_VIVO_CACHE = {}


def estado_en_vivo(fecha_iso):
    """Marcadores en vivo/finales desde StatsAPI, solo para MOSTRAR en la
    interfaz. Nunca participa en la verificacion (esa sigue usando
    dbo.GameLog con EsFinal=1). Se cachea 60 s para no martillar la API."""
    global _ULTIMA_CARGA_VIVO, _VIVO_CACHE
    ahora = time.time()
    if ahora - _ULTIMA_CARGA_VIVO < 60 and _VIVO_CACHE:
        return _VIVO_CACHE
    _ULTIMA_CARGA_VIVO = ahora
    url = ("https://statsapi.mlb.com/api/v1/schedule?sportId=1"
           f"&startDate={fecha_iso}&endDate={fecha_iso}")
    resultado = {}
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "MLB-Predictor/1.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            doc = json.load(resp)
        for fecha in doc.get("dates", []):
            for juego in fecha.get("games", []):
                local = juego["teams"]["home"]["team"]["name"]
                visita = juego["teams"]["away"]["team"]["name"]
                estado = juego["status"].get("abstractGameState", "")
                detallado = juego["status"].get("detailedState", "")
                hs = juego["teams"]["home"].get("score", 0) or 0
                vs = juego["teams"]["away"].get("score", 0) or 0
                resultado[(visita, local)] = {
                    "estado": estado, "detallado": detallado,
                    "local": hs, "visita": vs}
    except Exception:
        return {}
    _VIVO_CACHE = resultado
    return resultado


def estado_actual():
    """Datos para la interfaz: todas las apuestas (nuevas arriba) +
    totales absolutos de todo el historico."""
    hoy = date.today()
    resultado = {
        "fecha_hoy": hoy.isoformat(),
        "actualizado": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        "apuestas": [],
        "resumen": [],
        "totales": {"ganadas": 0, "perdidas": 0, "push": 0,
                    "pendientes": 0, "unidades": 0.0, "pl": 0.0},
        "ml": {"apuestas": [], "resumen": [],
               "totales": {"ganadas": 0, "perdidas": 0, "push": 0,
                           "pendientes": 0, "unidades": 0.0, "pl": 0.0}},
    }
    conexion = conectar()
    try:
        for r in conexion.execute(
            "SELECT p.Fecha, p.EquipoLocal, p.EquipoVisita, p.TipoApuesta, "
            "p.Linea, p.Unidades, p.Edge, p.Estado, p.CarrerasTotales, "
            "p.Cuota, g.HoraInicioUtc "
            "FROM dbo.Predicciones p "
            "LEFT JOIN dbo.GameLog g "
            "  ON g.Fecha = p.Fecha AND g.EquipoLocal = p.EquipoLocal "
            "  AND g.EquipoVisita = p.EquipoVisita "
            "ORDER BY p.Fecha DESC, p.Id DESC"
        ).fetchall():
            resultado["apuestas"].append({
                "fecha": r[0].isoformat(),
                "local": r[1],
                "visita": r[2],
                "tipo": r[3],
                "linea": float(r[4]),
                "unidades": float(r[5]),
                "edge": float(r[6]) if r[6] is not None else None,
                "estado": r[7],
                "total": int(r[8]) if r[8] is not None else None,
                "cuota": float(r[9]) if r[9] is not None else None,
                "hora": r[10].strftime("%Y-%m-%dT%H:%M:%SZ")
                        if r[10] is not None else None,
            })

        resumen_dias = {}
        for r in conexion.execute(
            "SELECT Fecha, Estado, Unidades, Cuota "
            "FROM dbo.Predicciones "
            "ORDER BY Fecha DESC, Estado"
        ).fetchall():
            fecha = r[0]
            estado = r[1]
            unidades = float(r[2])
            cuota = float(r[3]) if r[3] is not None else None
            pl = pl_jugada(estado, unidades, cuota)

            if estado == "GANADA":
                resultado["totales"]["ganadas"] += 1
            elif estado == "PERDIDA":
                resultado["totales"]["perdidas"] += 1
            elif estado == "PUSH":
                resultado["totales"]["push"] += 1
            elif estado == "PENDIENTE":
                resultado["totales"]["pendientes"] += 1
            elif estado == "NO_VALIDA":
                resultado["totales"].setdefault("no_validas", 0)
                resultado["totales"]["no_validas"] += 1
                continue  # no aporta unidades ni P/L: nunca fue apostable
            resultado["totales"]["unidades"] += unidades
            resultado["totales"]["pl"] += pl

            dia = resumen_dias.setdefault(fecha.isoformat(), {
                "fecha": fecha.isoformat(), "g": 0, "p": 0, "push": 0,
                "u": 0.0, "pl": 0.0})
            dia["u"] += unidades
            dia["pl"] += pl
            if estado == "GANADA":
                dia["g"] += 1
            elif estado == "PERDIDA":
                dia["p"] += 1
            elif estado == "PUSH":
                dia["push"] += 1

        for dia in resumen_dias.values():
            resultado["resumen"].append({
                "fecha": dia["fecha"], "g": dia["g"], "p": dia["p"],
                "push": dia["push"], "u": round(dia["u"], 2),
                "pl": round(dia["pl"], 2)})

        pendientes = [a for a in resultado["apuestas"] if a["estado"] == "PENDIENTE"]
        if pendientes:
            ahora_utc = datetime.now(timezone.utc)
            for fecha_iso in {a["fecha"] for a in pendientes}:
                vivo = estado_en_vivo(fecha_iso)
                for a in pendientes:
                    if a["fecha"] != fecha_iso:
                        continue
                    info = vivo.get((a["visita"], a["local"]))
                    if not info:
                        continue
                    if info["estado"] == "Live":
                        a["en_vivo"] = f"{info['visita']}-{info['local']}"
                    elif (info["estado"] == "Final"
                          and (info["local"] or info["visita"])):
                        a["final_por_api"] = f"{info['visita']}-{info['local']}"

        # ===== MERCADO MONEYLINE (aislado) =====
        for r in conexion.execute(
            "SELECT p.Fecha, p.EquipoLocal, p.EquipoVisita, p.TipoApuesta, "
            "p.Linea, p.Unidades, p.Edge, p.Estado, "
            "p.CarrerasLocal, p.CarrerasVisita, p.Cuota, p.ProbModelo "
            "FROM dbo.PrediccionesML p "
            "ORDER BY p.Fecha DESC, p.Id DESC"
        ).fetchall():
            resultado["ml"]["apuestas"].append({
                "fecha": r[0].isoformat(),
                "local": r[1],
                "visita": r[2],
                "tipo": r[3],
                "linea": float(r[4]) if r[4] is not None else None,
                "unidades": float(r[5]),
                "edge": float(r[6]) if r[6] is not None else None,
                "estado": r[7],
                "marcador": (f"{r[8]}-{r[9]}"
                             if r[8] is not None and r[9] is not None else None),
                "cuota": float(r[10]) if r[10] is not None else None,
                "prob": float(r[11]) if r[11] is not None else None,
            })

        resumen_ml_dias = {}
        for r in conexion.execute(
            "SELECT Fecha, Estado, Unidades, Cuota "
            "FROM dbo.PrediccionesML ORDER BY Fecha DESC, Estado"
        ).fetchall():
            fecha = r[0]
            estado = r[1]
            unidades = float(r[2])
            cuota = float(r[3]) if r[3] is not None else None
            pl = pl_jugada(estado, unidades, cuota)

            if estado == "GANADA":
                resultado["ml"]["totales"]["ganadas"] += 1
            elif estado == "PERDIDA":
                resultado["ml"]["totales"]["perdidas"] += 1
            elif estado == "PUSH":
                resultado["ml"]["totales"]["push"] += 1
            elif estado == "PENDIENTE":
                resultado["ml"]["totales"]["pendientes"] += 1
            resultado["ml"]["totales"]["unidades"] += unidades
            resultado["ml"]["totales"]["pl"] += pl

            dia = resumen_ml_dias.setdefault(fecha.isoformat(), {
                "fecha": fecha.isoformat(), "g": 0, "p": 0, "push": 0,
                "u": 0.0, "pl": 0.0})
            dia["u"] += unidades
            dia["pl"] += pl
            if estado == "GANADA":
                dia["g"] += 1
            elif estado == "PERDIDA":
                dia["p"] += 1
            elif estado == "PUSH":
                dia["push"] += 1

        for dia in resumen_ml_dias.values():
            resultado["ml"]["resumen"].append({
                "fecha": dia["fecha"], "g": dia["g"], "p": dia["p"],
                "push": dia["push"], "u": round(dia["u"], 2),
                "pl": round(dia["pl"], 2)})

        return resultado
    finally:
        conexion.close()


HTML = """<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pronosticos MLB</title>
<style>
  :root { --fondo:#000000; --tarjeta:#0c0c0c; --borde:#262626;
          --texto:#eaeaea; --verde:#00ff8c; --rojo:#ff3b3b;
          --ambar:#ffb020; --gris:#8a8a8a; --cian:#00e0ff; }
  * { box-sizing:border-box; margin:0; padding:0; }
  body { background:var(--fondo); color:var(--texto);
         font-family:Segoe UI, Arial, sans-serif;
         padding:clamp(12px, 2vw, 32px);
         background-image:radial-gradient(ellipse 80% 40% at 50% -10%,
             rgba(0,255,140,0.07), transparent); }
  .contenedor { width:100%; margin:0 auto; }
  h1 { font-size:clamp(26px, 3.2vw, 40px); margin-bottom:6px; font-weight:800;
       letter-spacing:0.5px; }
  h1::after { content:""; display:block; width:64px; height:4px;
              margin-top:8px; border-radius:2px;
              background:linear-gradient(90deg, var(--verde), var(--cian));
              box-shadow:0 0 12px rgba(0,255,140,0.6); }
  .sub { color:var(--gris); font-size:clamp(13px, 1.4vw, 15px); margin-bottom:24px; }
  .card { background:var(--tarjeta); border:1px solid var(--borde);
          border-radius:12px; padding:clamp(14px, 1.6vw, 24px);
          box-shadow:0 8px 28px rgba(0,0,0,0.5); }
  .card h2 { font-size:clamp(17px, 1.8vw, 22px); margin-bottom:14px; color:#f5f5f5;
             border-left:4px solid var(--verde); padding-left:10px;
             font-weight:700; }
  .grid { display:grid; grid-template-columns:repeat(auto-fit,minmax(min(460px,100%),1fr));
          gap:clamp(14px, 1.6vw, 20px); margin-top:20px; }
  .grupo { margin-bottom:14px; }
  .grupo:last-child { margin-bottom:0; }
  .grupo-fecha { display:flex; justify-content:space-between;
                 align-items:center; flex-wrap:wrap; gap:8px;
                 font-size:clamp(15px, 1.5vw, 18px); font-weight:700; color:#ffffff;
                 padding:8px 12px; border-bottom:2px solid var(--borde);
                 border-left:3px solid var(--cian);
                 border-radius:8px 8px 0 0;
                 background:linear-gradient(90deg, rgba(0,224,255,0.06), transparent);
                 letter-spacing:0.4px; }
  .grupo-resumen { font-size:clamp(13px, 1.3vw, 15px); font-weight:600; color:var(--gris); }
  .grupo-resumen b { font-size:clamp(14px, 1.4vw, 16px); }
  .partido { display:flex; justify-content:space-between; align-items:center;
             padding:13px 4px; border-bottom:1px dashed #1f1f1f;
             transition:background 0.15s; border-radius:6px; }
  .partido:hover { background:rgba(255,255,255,0.03); }
  .partido:last-child { border-bottom:none; }
  .equipos { font-size:clamp(16px, 1.7vw, 21px); font-weight:600; }
  .detalle { color:var(--gris); font-size:clamp(13px, 1.3vw, 15px); margin-top:4px; }
  .detalle .vivo { color:var(--cian); font-weight:700;
                   text-shadow:0 0 8px rgba(0,224,255,0.4); }
  .derecha { text-align:right; }
  .badge { display:inline-block; padding:5px 14px; border-radius:999px;
           font-size:clamp(13px, 1.3vw, 15px); font-weight:800; letter-spacing:0.5px; }
  .b-pendiente { background:rgba(255,176,32,0.08); color:var(--ambar);
                 border:1px solid var(--ambar);
                 box-shadow:0 0 10px rgba(255,176,32,0.25); }
  .b-ganada { background:rgba(0,255,140,0.08); color:var(--verde);
              border:1px solid var(--verde);
              box-shadow:0 0 12px rgba(0,255,140,0.35); }
  .b-perdida { background:rgba(255,59,59,0.08); color:var(--rojo);
               border:1px solid var(--rojo);
               box-shadow:0 0 12px rgba(255,59,59,0.3); }
  .b-push { background:rgba(138,138,138,0.08); color:var(--gris);
            border:1px solid var(--gris); }
  .total { font-size:clamp(15px, 1.6vw, 19px); margin-top:6px; line-height:1.7; }
  .total b { font-size:clamp(17px, 1.8vw, 21px); }
  .pl-pos { color:var(--verde); font-weight:700;
            text-shadow:0 0 8px rgba(0,255,140,0.4); }
  .pl-neg { color:var(--rojo); font-weight:700;
            text-shadow:0 0 8px rgba(255,59,59,0.4); }
  .pl-cero { color:var(--gris); }
  .vacio { color:var(--gris); font-size:15px; }
  .tabla-scroll { max-height:520px; overflow-y:auto; border-radius:8px; }
  .tabla-scroll::-webkit-scrollbar { width:10px; }
  .tabla-scroll::-webkit-scrollbar-track { background:#0a0a0a; }
  .tabla-scroll::-webkit-scrollbar-thumb { background:#2a2a2a;
      border-radius:5px; }
  .tabla-scroll::-webkit-scrollbar-thumb:hover { background:var(--verde); }
  table { width:100%; border-collapse:collapse; font-size:clamp(13px, 1.4vw, 16px); }
  th, td { text-align:left; padding:9px 12px; border-bottom:1px solid #1f1f1f; }
  th { color:var(--gris); font-weight:700; text-transform:uppercase;
       font-size:clamp(11px, 1.2vw, 13px); letter-spacing:1px; position:sticky; top:0;
       background:#0c0c0c; }
  tr:hover td { background:rgba(255,255,255,0.03); }
  .pie { color:var(--gris); font-size:13px; margin-top:20px; text-align:center; }
  @media (max-width: 640px) {
    .partido { flex-wrap:wrap; gap:6px; }
    .grupo-fecha { flex-direction:column; align-items:flex-start; }
    .tabla-scroll { max-height:360px; }
  }
</style>
</head>
<body>
<div class="contenedor">
  <h1>Pronosticos MLB</h1>
  <div class="sub">Actualizado: <span id="actualizado">...</span>
      (la pagina se refresca sola cada 30 s)</div>

  <div class="card">
    <h2 id="titulo_apuestas">Apuestas</h2>
    <div id="apuestas"></div>
  </div>

  <div class="grid">
    <div class="card">
      <h2>Totales historicos (absoluto)</h2>
      <div id="totales" class="total"></div>
    </div>
    <div class="card">
      <h2>Resumen diario</h2>
      <div class="tabla-scroll">
      <table>
        <thead><tr><th>Dia</th><th>G</th><th>P</th><th>Push</th>
               <th>u apost.</th><th>P/L</th></tr></thead>
        <tbody id="resumen"></tbody>
      </table>
      </div>
    </div>
  </div>

  <div class="card" style="margin-top:20px; border-color:var(--cian);">
    <h2 style="border-left-color:var(--cian);">Mercado Moneyline (ganador)</h2>
    <div id="apuestas_ml"></div>
    <div class="grid" style="margin-top:16px;">
      <div class="card">
        <h2 style="border-left-color:var(--cian);">Totales ML (absoluto)</h2>
        <div id="totales_ml" class="total"></div>
      </div>
      <div class="card">
        <h2 style="border-left-color:var(--cian);">Resumen diario ML</h2>
        <div class="tabla-scroll">
        <table>
          <thead><tr><th>Dia</th><th>G</th><th>P</th><th>Push</th>
                 <th>u apost.</th><th>P/L</th></tr></thead>
          <tbody id="resumen_ml"></tbody>
        </table>
        </div>
      </div>
    </div>
  </div>
  <div class="pie">Local: solo esta laptop (localhost). Fuente:
      dbo.Predicciones + dbo.PrediccionesML + dbo.GameLog de MLB_Predictive.</div>
</div>

<script>
  const ESTADOS = {
    "PENDIENTE": ["Pendiente", "b-pendiente"],
    "GANADA": ["Ganada", "b-ganada"],
    "PERDIDA": ["Perdida", "b-perdida"],
    "PUSH": ["Push", "b-push"],
    "NO_VALIDA": ["No válida", "b-pendiente"],
  };
  const FALLBACK_PAGO = 100.0 / 110.0;
  function badge(estado) {
    const [texto, clase] = ESTADOS[estado] || [estado, "b-push"];
    return `<span class="badge ${clase}">${texto}</span>`;
  }
  function plClass(pl) {
    return pl > 0 ? "pl-pos" : (pl < 0 ? "pl-neg" : "pl-cero");
  }
  function fmtFecha(iso) {
    const [a, m, d] = iso.split("-");
    return `${d}-${m}-${a}`;
  }
  function fmtHora(iso) {
    try {
      return new Date(iso).toLocaleTimeString([], {hour: "2-digit", minute: "2-digit"});
    } catch (e) { return ""; }
  }
  function plJugada(p) {
    if (p.estado === "GANADA")
      return p.unidades * ((p.cuota !== null ? p.cuota : FALLBACK_PAGO + 1) - 1);
    if (p.estado === "PERDIDA") return -p.unidades;
    return 0;
  }
  function pintarApuestas(lista) {
    const cont = document.getElementById("apuestas");
    const titulo = document.getElementById("titulo_apuestas");
    if (!lista.length) {
      titulo.textContent = "Apuestas de hoy";
      cont.innerHTML = '<div class="vacio">Aun no hay pronosticos registrados.</div>';
      return;
    }
    const primera = lista[0].fecha;
    titulo.textContent = primera === fechaActual
      ? "Apuestas de hoy" : `Apuestas del ${fmtFecha(primera)}`;
    const grupos = {};
    for (const p of lista) (grupos[p.fecha] = grupos[p.fecha] || []).push(p);
    let html = "";
    let idx = 0;
    for (const [fecha, listaDia] of Object.entries(grupos)) {
      let g = 0, per = 0, push = 0, pl = 0;
      for (const p of listaDia) {
        if (p.estado === "GANADA") { g++; pl += plJugada(p); }
        else if (p.estado === "PERDIDA") { per++; pl += plJugada(p); }
        else if (p.estado === "PUSH") push++;
      }
      const rotulo = idx === 0 ? "" : `${fmtFecha(fecha)}`;
      html += `<div class="grupo">
        <div class="grupo-fecha">${rotulo}
          <span class="grupo-resumen">G ${g} | P ${per} | Push ${push} |
            <b class="${plClass(pl)}">${pl > 0 ? "+" : ""}${pl.toFixed(2)} u</b></span>
        </div>`;
      for (const p of listaDia) {
        const total = p.total !== null ? ` | total real: ${p.total}` : "";
        const edge = p.edge !== null ? ` | edge: ${p.edge.toFixed(2)}` : "";
        const cuota = p.cuota !== null ? ` | cuota: ${p.cuota.toFixed(2)}` : "";
        const hora = p.hora !== null ? ` | inicio: ${fmtHora(p.hora)}` : "";
        const vivo = p.en_vivo !== undefined
          ? ` | <span class="vivo">EN CURSO: ${p.en_vivo}</span>` : "";
        const finalApi = p.final_por_api !== undefined
          ? ` | final: ${p.final_por_api} (pend. verificar)` : "";
        html += `<div class="partido">
          <div>
            <div class="equipos">${p.local} vs ${p.visita}</div>
            <div class="detalle">${p.unidades.toFixed(2)} u - ${p.tipo} a la linea ${p.linea.toFixed(1)}${hora}${vivo}${finalApi}${cuota}${edge}${total}</div>
          </div>
          <div class="derecha">${badge(p.estado)}</div>
        </div>`;
      }
      html += "</div>";
      idx++;
    }
    cont.innerHTML = html;
  }
  function pintarTotales(t) {
    const el = document.getElementById("totales");
    const pl = t.pl;
    const clase = plClass(pl);
    el.innerHTML = `G: <b class="pl-pos">${t.ganadas}</b> |
        P: <b class="pl-neg">${t.perdidas}</b> |
        Push: ${t.push} | Pendientes: ${t.pendientes}<br>
        Unidades apostadas: ${t.unidades.toFixed(2)} |
        P/L: <span class="${clase}">${pl > 0 ? "+" : ""}${pl.toFixed(2)} u</span> (cuota real)`;
  }
  function pintarResumen(resumen) {
    const filas = document.getElementById("resumen");
    let html = "";
    for (const d of resumen) {
      const clase = plClass(d.pl);
      html += `<tr><td>${fmtFecha(d.fecha)}</td><td>${d.g}</td><td>${d.p}</td>
        <td>${d.push}</td><td>${d.u.toFixed(2)}</td>
        <td class="${clase}">${d.pl > 0 ? "+" : ""}${d.pl.toFixed(2)}</td></tr>`;
    }
    filas.innerHTML = html || '<tr><td colspan="6" class="vacio">Sin historial aun</td></tr>';
  }
  function pintarApuestasML(lista) {
    const cont = document.getElementById("apuestas_ml");
    if (!lista.length) {
      cont.innerHTML = '<div class="vacio">Aun no hay pronosticos ML registrados.</div>';
      return;
    }
    function apodo(nombre) {
      const partes = nombre.trim().split(" ");
      return partes[partes.length - 1].toUpperCase();
    }
    let html = "";
    for (const p of lista) {
      const ganador = p.tipo === "HOME" ? p.local : p.visita;
      const detalle = [];
      detalle.push(`${p.local} vs ${p.visita}`);
      if (p.cuota !== null) detalle.push(`cuota: ${p.cuota.toFixed(2)}`);
      if (p.prob !== null) detalle.push(`P: ${(p.prob * 100).toFixed(0)}%`);
      if (p.edge !== null) detalle.push(`edge: ${(p.edge * 100).toFixed(1)} pts`);
      if (p.marcador !== null) detalle.push(`final: ${p.marcador}`);
      const p_registro = p.unidades.toFixed(2);
      html += `<div class="partido" style="flex-direction:column; align-items:stretch; gap:4px;">
        <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:8px;">
          <div class="equipos"><span style="font-weight:800;">GANA ${apodo(ganador)}</span>
            <span class="detalle" style="display:inline; margin-left:10px; color:var(--gris); font-weight:400;">
              ${fmtFecha(p.fecha)} | ${p_registro} u</span></div>
          <div class="derecha">${badge(p.estado)}</div>
        </div>
        <div class="detalle">${detalle.join(" | ")}</div>
      </div>`;
    }
    cont.innerHTML = html;
  }
  function pintarTotalesML(t) {
    const el = document.getElementById("totales_ml");
    const pl = t.pl;
    const clase = plClass(pl);
    el.innerHTML = `G: <b class="pl-pos">${t.ganadas}</b> |
        P: <b class="pl-neg">${t.perdidas}</b> |
        Push: ${t.push} | Pendientes: ${t.pendientes}<br>
        Unidades apostadas: ${t.unidades.toFixed(2)} |
        P/L: <span class="${clase}">${pl > 0 ? "+" : ""}${pl.toFixed(2)} u</span> (cuota real)`;
  }
  function pintarResumenML(resumen) {
    const filas = document.getElementById("resumen_ml");
    let html = "";
    for (const d of resumen) {
      const clase = plClass(d.pl);
      html += `<tr><td>${fmtFecha(d.fecha)}</td><td>${d.g}</td><td>${d.p}</td>
        <td>${d.push}</td><td>${d.u.toFixed(2)}</td>
        <td class="${clase}">${d.pl > 0 ? "+" : ""}${d.pl.toFixed(2)}</td></tr>`;
    }
    filas.innerHTML = html || '<tr><td colspan="6" class="vacio">Sin historial ML aun</td></tr>';
  }
  let fechaActual = "";
  async function cargar() {
    try {
      const resp = await fetch("/api");
      const data = await resp.json();
      fechaActual = data.fecha_hoy;
      document.getElementById("actualizado").textContent = data.actualizado;
      pintarApuestas(data.apuestas);
      pintarTotales(data.totales);
      pintarResumen(data.resumen);
      pintarApuestasML(data.ml.apuestas);
      pintarTotalesML(data.ml.totales);
      pintarResumenML(data.ml.resumen);
    } catch (e) {
      document.getElementById("actualizado").textContent = "Servidor no responde aun...";
    }
  }
  cargar();
  setInterval(cargar, 30000);
</script>
</body>
</html>
"""


class Manejador(BaseHTTPRequestHandler):
    def do_GET(self):
        ruta = self.path.split("?")[0]
        if ruta in ("/", "/index.html"):
            cuerpo = HTML.encode("utf-8")
            self._responder(200, cuerpo, "text/html; charset=utf-8")
        elif ruta == "/api":
            try:
                cuerpo = json.dumps(estado_actual(),
                                    ensure_ascii=False).encode("utf-8")
                self._responder(200, cuerpo, "application/json; charset=utf-8")
            except Exception as ex:
                cuerpo = json.dumps({"error": str(ex)}).encode("utf-8")
                self._responder(500, cuerpo, "application/json; charset=utf-8")
        else:
            self._responder(404, b"Not found", "text/plain; charset=utf-8")

    def _responder(self, codigo, cuerpo, tipo):
        self.send_response(codigo)
        self.send_header("Content-Type", tipo)
        self.send_header("Content-Length", str(len(cuerpo)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(cuerpo)

    def log_message(self, formato, *args):
        pass


def main():
    servidor = ThreadingHTTPServer(("127.0.0.1", PUERTO), Manejador)
    print(f"[WEB] Pronosticos MLB en http://localhost:{PUERTO} "
          "(Ctrl+C para detener)")
    try:
        servidor.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        servidor.server_close()


if __name__ == "__main__":
    main()
