# src/envs/disease_control_env/server/app.py
import os
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from core.env_server import create_fastapi_app, create_web_interface_app
from ..models import DiseaseAction, DiseaseObservation
from .disease_control_environment import DiseaseControlEnvironment

env = DiseaseControlEnvironment()
app: FastAPI = create_fastapi_app(env, DiseaseAction, DiseaseObservation)

if __name__ == "__main__":
  import uvicorn
  uvicorn.run(app, host="0.0.0.0", port=8000)

@app.get("/grid")
def grid():
    return env.get_grid()

# ---- JSON feeds used by the mini dashboard ----
@app.get("/timeseries", response_class=JSONResponse)
def timeseries(tail: int | None = Query(default=400, ge=1)):
    return env.get_timeseries(tail=tail)

@app.get("/events", response_class=JSONResponse)
def events():
    return {"events": env.events}

# ---- Minimal web UI (no external build, uses Chart.js CDN) ----
ENABLE_WEB = os.getenv("ENABLE_WEB_INTERFACE", "true").lower() == "true"

if ENABLE_WEB:
    @app.get("/web", response_class=HTMLResponse)
    def web_ui():
        return HTML_TEMPLATE

HTML_TEMPLATE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Disease Control · Live</title>
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <style>
    body{font-family:ui-sans-serif,system-ui,Segoe UI,Roboto,Arial;margin:20px;color:#111}
    h1{margin:0 0 10px}
    .row{display:flex;gap:24px;flex-wrap:wrap}
    .card{flex:1 1 480px;border:1px solid #ddd;border-radius:12px;padding:16px;box-shadow:0 2px 8px rgba(0,0,0,.05)}
    canvas{width:100%;height:320px}
    .pill{display:inline-block;padding:2px 10px;border-radius:999px;background:#f2f2f2;margin-left:8px}
    .events{max-height:140px;overflow:auto;font-family:ui-monospace,monospace;background:#fafafa;border:1px dashed #ddd;border-radius:8px;padding:8px}
    .mut {color:#b00020;font-weight:600}
    .ref {color:#0b6a00;font-weight:600}
  </style>
</head>
<body>
  <h1>Disease Control <span id="disease" class="pill"></span></h1>
  <div class="row">
    <div class="card" style="margin-top:24px">
  <h3>Person Grid (32×32)</h3>
  <canvas id="gridCanvas" width="320" height="320" style="image-rendering: pixelated; border:1px solid #ddd; border-radius:8px"></canvas>
  <div style="margin-top:8px;font-family:ui-monospace">
    Legend:
    <span style="color:#1f77b4">■</span> S
    <span style="color:#2ca02c">■</span> H
    <span style="color:#d62728">■</span> I
    <span style="color:#9467bd">■</span> Q
    <span style="color:#8c564b">■</span> D
  </div>
</div>
    <div class="card"><h3>Epidemic Curves (S,H,I,Q,D)</h3><canvas id="chart1"></canvas></div>
    <div class="card"><h3>Budget</h3><canvas id="chart2"></canvas></div>
  </div>
  <div class="card" style="margin-top:24px">
    <h3>Events</h3>
    <div id="events" class="events"></div>
  </div>
  <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
  <script>
    const q = (s)=>document.querySelector(s);
    const diseaseEl = q("#disease");
    let chart1, chart2;

    function mkChart(ctx, labels, datasets){
      return new Chart(ctx, {
        type: 'line',
        data: { labels, datasets },
        options: {
          responsive:true,
          animation:false,
          interaction:{ mode:'index', intersect:false },
          scales:{ x:{ title:{ display:true, text:'step'} } }
        }
      });
    }

    async function fetchTS(){
      const r = await fetch('/timeseries?tail=400');
      return await r.json();
    }
    function color(idx){ // stable palette
      const cs = ['#1f77b4','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2'];
      return cs[idx % cs.length];
    }

    async function tick(){
      const ts = await fetchTS();
      const lbls = ts.step || [];
      diseaseEl.textContent = (window._disease || 'preset');
      const ds1 = [
        {label:'S', data:ts.S, borderColor:color(0), fill:false},
        {label:'H', data:ts.H, borderColor:color(1), fill:false},
        {label:'I', data:ts.I, borderColor:color(2), fill:false},
        {label:'Q', data:ts.Q, borderColor:color(3), fill:false},
        {label:'D', data:ts.D, borderColor:color(4), fill:false},
      ];
      const ds2 = [{label:'Budget', data:ts.budget, borderColor:color(5), fill:false}];

      if(!chart1){
        chart1 = mkChart(document.getElementById('chart1'), lbls, ds1);
        chart2 = mkChart(document.getElementById('chart2'), lbls, ds2);
      } else {
        chart1.data.labels = lbls; chart1.data.datasets = ds1; chart1.update();
        chart2.data.labels = lbls; chart2.data.datasets = ds2; chart2.update();
      }
      // events
      const evDiv = q('#events');
      evDiv.innerHTML = (ts.events || []).slice(-30).map(e=>{
        const cls = e.includes('mutation') ? 'mut' : (e.includes('refill') ? 'ref' : '');
        return `<div class="${cls}">${e}</div>`;
      }).join('');
    }

    setInterval(tick, 800);
    tick();
    const gridColors = {
      0: '#1f77b4', // S
      1: '#2ca02c', // H
      2: '#d62728', // I
      3: '#9467bd', // Q
      4: '#8c564b', // D
    };

    async function drawGrid(){
      const r = await fetch('/grid');
      const data = await r.json();
      if(!data.enabled) return;
      const N = data.size || 32;
      const grid = data.grid || [];
      const cvs = document.getElementById('gridCanvas');
      const ctx = cvs.getContext('2d');
      const cell = Math.floor(cvs.width / N);
      ctx.clearRect(0,0,cvs.width,cvs.height);
      for(let i=0;i<N;i++){
        for(let j=0;j<N;j++){
          const s = grid[i][j];
          ctx.fillStyle = gridColors[s] || '#999';
          ctx.fillRect(j*cell, i*cell, cell, cell);
        }
      }
    }

    setInterval(drawGrid, 500);
  </script>
</body>
</html>
"""
