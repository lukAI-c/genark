#!/usr/bin/env python3
"""
Genark Web Broadcaster — 嵌入式实时 Web 可视化模块

用法:
    from genark_web_viewer import WebBroadcaster

    broadcaster = WebBroadcaster(output_dir="web_output", port=9998)
    ...
    broadcaster.send_frame(rgb_array, metadata={"step": i})
    ...
    broadcaster.finish()
    broadcaster.wait_for_exit()  # 可选, 保持 HTTP 服务用于回放
"""

import json
import os
import signal
import subprocess
import threading
import time
from http.server import HTTPServer, SimpleHTTPRequestHandler

import numpy as np


# ============================================================
# Image saving (auto-detect available library)
# ============================================================
def _make_save_fn():
    try:
        from PIL import Image

        def _save(arr, path, quality=85):
            Image.fromarray(arr).save(path, "JPEG", quality=quality)

        return _save
    except ImportError:
        pass

    try:
        import cv2

        def _save(arr, path, quality=85):
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, quality])

        return _save
    except ImportError:
        pass

    try:
        import imageio

        def _save(arr, path, quality=85):
            imageio.imwrite(path, arr, quality=quality)

        return _save
    except ImportError:
        pass

    raise RuntimeError("需要 Pillow / OpenCV / imageio: pip install Pillow")


_save_image = _make_save_fn()


# ============================================================
# HTTP Server with ETag
# ============================================================
class _ETagHandler(SimpleHTTPRequestHandler):
    def send_head(self):
        path = self.translate_path(self.path)
        if os.path.isdir(path):
            return super().send_head()
        try:
            f = open(path, "rb")
        except OSError:
            self.send_error(404, "File not found")
            return None

        fs = os.fstat(f.fileno())
        etag = f'W/"{fs.st_mtime_ns:x}-{fs.st_size:x}"'
        inm = self.headers.get("If-None-Match")

        if inm and inm == etag:
            f.close()
            self.send_response(304)
            self.send_header("ETag", etag)
            self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Expose-Headers", "ETag")
            self.end_headers()
            return None

        self.send_response(200)
        ctype = self.guess_type(path)
        self.send_header("Content-type", ctype)
        self.send_header("Content-Length", str(fs.st_size))
        self.send_header("Last-Modified", self.date_time_string(fs.st_mtime))
        self.send_header("ETag", etag)
        self.send_header("Cache-Control", "public, max-age=0, must-revalidate")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Expose-Headers", "ETag")
        self.end_headers()
        return f

    def log_message(self, _fmt, *_args):
        pass


def _kill_port(port):
    """Release a port if occupied by another process."""
    try:
        result = subprocess.run(
            ["ss", "-tlnp"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            if f":{port}" in line and "pid=" in line:
                pid_str = line.split("pid=")[1].split(",")[0].split(")")[0]
                pid = int(pid_str)
                if pid != os.getpid():
                    print(f"[WebViewer] 释放端口 {port} (kill PID {pid})")
                    os.kill(pid, signal.SIGTERM)
                    time.sleep(0.5)
    except Exception:
        pass


# ============================================================
# Embedded index.html
# ============================================================
_INDEX_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Genark Web Viewer</title>
<style>
*{margin:0;padding:0;box-sizing:border-box}
body{background:#0d1117;color:#c9d1d9;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,'Noto Sans SC',sans-serif;min-height:100vh;display:flex;flex-direction:column;}
header{background:#161b22;border-bottom:1px solid #30363d;padding:12px 24px;display:flex;justify-content:space-between;align-items:center;flex-shrink:0;}
header h1{font-size:18px;font-weight:600;letter-spacing:.5px}
header h1 .accent{color:#58a6ff}
.badges{display:flex;gap:12px;font-size:12px;align-items:center;flex-wrap:wrap}
.badge{display:flex;align-items:center;gap:5px;background:#21262d;padding:4px 10px;border-radius:12px;border:1px solid #30363d;}
.badge strong{color:#e6edf3}
.dot{width:7px;height:7px;border-radius:50%;flex-shrink:0}
.dot-wait{background:#d29922;animation:pulse 1.2s infinite}
.dot-live{background:#3fb950;animation:pulse 1.5s infinite}
.dot-done{background:#58a6ff}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}
.main{flex:1;display:flex;flex-direction:column;align-items:center;padding:16px 24px;overflow:hidden}
.canvas-wrap{position:relative;flex:1;display:flex;align-items:center;justify-content:center;width:100%;max-width:1100px;min-height:0;}
#render-view{max-width:100%;max-height:100%;border-radius:6px;border:1px solid #30363d;background:#0d1117;object-fit:contain;display:none;}
.placeholder{width:960px;max-width:100%;aspect-ratio:4/3;border-radius:6px;border:2px dashed #30363d;display:flex;flex-direction:column;align-items:center;justify-content:center;gap:12px;color:#484f58;}
.placeholder svg{width:48px;height:48px;opacity:.25}
.placeholder p:first-of-type{font-size:15px;color:#8b949e}
.placeholder p:last-of-type{font-size:12px}
.info-panel{width:100%;max-width:1100px;margin-top:8px;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 16px;font-size:13px;color:#8b949e;flex-shrink:0;display:none;}
.info-panel .info-row{display:flex;gap:24px;flex-wrap:wrap}
.info-panel .info-item{display:flex;gap:6px;align-items:baseline}
.info-panel .info-label{color:#484f58;font-size:11px;text-transform:uppercase;letter-spacing:.5px}
.info-panel .info-value{color:#c9d1d9;font-weight:500}
.info-panel .info-instruction{margin-top:6px;color:#c9d1d9;font-style:italic;line-height:1.4;max-height:3em;overflow:hidden;text-overflow:ellipsis}
.controls{width:100%;max-width:1100px;margin-top:8px;flex-shrink:0;background:#161b22;border:1px solid #30363d;border-radius:8px;padding:10px 16px;display:flex;align-items:center;gap:10px;}
.btn-group{display:flex;gap:0}
.btn-group button{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:5px 14px;font-size:12px;cursor:pointer;transition:all .15s;}
.btn-group button:first-child{border-radius:6px 0 0 6px}
.btn-group button:last-child{border-radius:0 6px 6px 0;border-left:0}
.btn-group button:hover{background:#30363d}
.btn-group button.active{background:#1f6feb;border-color:#1f6feb;color:#fff}
#frame-slider{flex:1;accent-color:#58a6ff;height:4px}
#frame-label{font-size:12px;font-variant-numeric:tabular-nums;min-width:90px;text-align:right;color:#8b949e;}
.speed-select{background:#21262d;border:1px solid #30363d;color:#c9d1d9;padding:4px 8px;border-radius:4px;font-size:12px;cursor:pointer;}
.icon-btn{background:none;border:none;color:#8b949e;cursor:pointer;padding:4px;border-radius:4px;display:flex;align-items:center;}
.icon-btn:hover{color:#c9d1d9;background:#21262d}
footer{padding:8px;text-align:center;font-size:11px;color:#30363d;flex-shrink:0;}
</style>
</head>
<body>
<header>
  <h1><span class="accent">Genark</span> Web Viewer</h1>
  <div class="badges">
    <div class="badge"><div class="dot dot-wait" id="dot"></div><span id="conn-text">等待连接</span></div>
    <div class="badge">帧 <strong id="frame-count">-/-</strong></div>
    <div class="badge">FPS <strong id="fps-val">-</strong></div>
  </div>
</header>
<div class="main">
  <div class="canvas-wrap">
    <div class="placeholder" id="placeholder">
      <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
        <path d="M15.75 10.5l4.72-4.72a.75.75 0 011.28.53v11.38a.75.75 0 01-1.28.53l-4.72-4.72M4.5 18.75h9a2.25 2.25 0 002.25-2.25v-9A2.25 2.25 0 0013.5 5.25h-9A2.25 2.25 0 002.25 7.5v9A2.25 2.25 0 004.5 18.75z"/>
      </svg>
      <p>等待仿真数据...</p>
      <p>Genark WebBroadcaster</p>
    </div>
    <img id="render-view" alt="Render">
  </div>
  <div class="info-panel" id="info-panel">
    <div class="info-row">
      <div class="info-item"><span class="info-label">Episode</span><span class="info-value" id="info-episode">-</span></div>
      <div class="info-item"><span class="info-label">Step</span><span class="info-value" id="info-step">-</span></div>
      <div class="info-item"><span class="info-label">Action</span><span class="info-value" id="info-action">-</span></div>
      <div class="info-item"><span class="info-label">Active</span><span class="info-value" id="info-active">-</span></div>
      <div class="info-item"><span class="info-label">Dist</span><span class="info-value" id="info-dist">-</span></div>
    </div>
    <div class="info-instruction" id="info-instruction"></div>
  </div>
  <div class="controls">
    <div class="btn-group">
      <button id="btn-live" class="active" title="实时跟踪最新帧">实时</button>
      <button id="btn-play" title="自动回放所有帧">回放</button>
    </div>
    <button class="icon-btn" id="btn-prev" title="上一帧">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M9.78 12.78a.75.75 0 01-1.06 0L4.47 8.53a.75.75 0 010-1.06l4.25-4.25a.75.75 0 011.06 1.06L6.06 8l3.72 3.72a.75.75 0 010 1.06z"/></svg>
    </button>
    <button class="icon-btn" id="btn-next" title="下一帧">
      <svg width="16" height="16" viewBox="0 0 16 16" fill="currentColor"><path d="M6.22 3.22a.75.75 0 011.06 0l4.25 4.25a.75.75 0 010 1.06l-4.25 4.25a.75.75 0 01-1.06-1.06L9.94 8 6.22 4.28a.75.75 0 010-1.06z"/></svg>
    </button>
    <input type="range" id="frame-slider" min="0" max="0" value="0" disabled>
    <span id="frame-label">- / -</span>
    <select class="speed-select" id="speed-select" title="回放速度">
      <option value="200">5 fps</option>
      <option value="100">10 fps</option>
      <option value="50" selected>20 fps</option>
      <option value="33">30 fps</option>
      <option value="16">60 fps</option>
    </select>
  </div>
</div>
<footer>Genark Web Viewer &middot; 嵌入式实时可视化</footer>
<script>
(function(){
  const img=document.getElementById('render-view'),ph=document.getElementById('placeholder'),
    slider=document.getElementById('frame-slider'),frameLbl=document.getElementById('frame-label'),
    dot=document.getElementById('dot'),connText=document.getElementById('conn-text'),
    frameCnt=document.getElementById('frame-count'),fpsVal=document.getElementById('fps-val'),
    btnLive=document.getElementById('btn-live'),btnPlay=document.getElementById('btn-play'),
    btnPrev=document.getElementById('btn-prev'),btnNext=document.getElementById('btn-next'),
    speedSel=document.getElementById('speed-select'),
    infoPanel=document.getElementById('info-panel'),
    infoEpisode=document.getElementById('info-episode'),
    infoStep=document.getElementById('info-step'),
    infoAction=document.getElementById('info-action'),
    infoActive=document.getElementById('info-active'),
    infoDist=document.getElementById('info-dist'),
    infoInstruction=document.getElementById('info-instruction');

  let mode='live',lastETag=null,totalFrames=0,simRunning=false,playbackTimer=null,blobCache={};
  const MAX_CACHE=80;

  function showImage(url){
    const old=img.src;img.src=url;img.style.display='block';ph.style.display='none';
    if(old&&old.startsWith('blob:')&&old!==url&&!Object.values(blobCache).includes(old))URL.revokeObjectURL(old);
  }
  function setMode(m){
    mode=m;btnLive.classList.toggle('active',m==='live');btnPlay.classList.toggle('active',m==='playback');
    slider.disabled=(m==='live');if(m==='live')stopPlayback();
  }
  function stopPlayback(){if(playbackTimer){clearInterval(playbackTimer);playbackTimer=null;}btnPlay.textContent='回放';}
  function startPlayback(){
    if(totalFrames<=0)return;const ms=parseInt(speedSel.value)||50;slider.value=0;
    playbackTimer=setInterval(async()=>{
      let idx=parseInt(slider.value);if(idx>=parseInt(slider.max)){stopPlayback();return;}
      idx++;slider.value=idx;await loadFrame(idx);
    },ms);btnPlay.textContent='停止';
  }
  btnLive.onclick=()=>setMode('live');
  btnPlay.onclick=()=>{if(mode==='playback'&&playbackTimer){stopPlayback();setMode('live');return;}setMode('playback');startPlayback();};
  btnPrev.onclick=()=>{if(mode==='live')setMode('playback');stopPlayback();const idx=Math.max(0,parseInt(slider.value)-1);slider.value=idx;loadFrame(idx);};
  btnNext.onclick=()=>{if(mode==='live')setMode('playback');stopPlayback();const idx=Math.min(parseInt(slider.max),parseInt(slider.value)+1);slider.value=idx;loadFrame(idx);};
  slider.oninput=()=>{if(mode!=='playback')setMode('playback');stopPlayback();loadFrame(parseInt(slider.value));};

  async function loadFrame(idx){
    const padded=String(idx).padStart(6,'0'),url=`frames/frame_${padded}.jpg`;
    frameLbl.textContent=`${idx+1} / ${totalFrames}`;
    if(blobCache[idx]){showImage(blobCache[idx]);return;}
    try{const r=await fetch(url);if(!r.ok)return;const blob=await r.blob(),burl=URL.createObjectURL(blob);
      if(Object.keys(blobCache).length>MAX_CACHE){const oldest=Object.keys(blobCache)[0];URL.revokeObjectURL(blobCache[oldest]);delete blobCache[oldest];}
      blobCache[idx]=burl;showImage(burl);
    }catch(_){}
  }

  function updateInfo(d){
    const m=d.metadata;
    if(!m){infoPanel.style.display='none';return;}
    infoPanel.style.display='block';
    infoEpisode.textContent=m.episode_id||m.episode||'-';
    infoStep.textContent=m.step!=null?m.step:'-';
    infoAction.textContent=m.action||'-';
    infoActive.textContent=m.active!=null?`${m.active}/${m.total_envs||'?'}`:'-';
    infoDist.textContent=m.distance_to_goal!=null?parseFloat(m.distance_to_goal).toFixed(2):'-';
    if(m.instruction){infoInstruction.textContent=m.instruction;infoInstruction.style.display='block';}
    else{infoInstruction.style.display='none';}
  }

  async function pollStatus(){
    try{const r=await fetch('status.json?_='+Date.now());if(!r.ok)throw 0;const d=await r.json();
      totalFrames=d.frame;simRunning=d.running;
      frameCnt.textContent=`${d.frame}/${d.total>0?d.total:'\u221e'}`;
      fpsVal.textContent=d.fps;slider.max=Math.max(0,d.frame-1);
      if(simRunning){dot.className='dot dot-live';connText.textContent='\u4eff\u771f\u4e2d';}
      else{dot.className='dot dot-done';connText.textContent='\u5df2\u5b8c\u6210';}
      updateInfo(d);
    }catch(_){dot.className='dot dot-wait';connText.textContent='\u7b49\u5f85\u8fde\u63a5';}
  }

  async function pollLive(){
    if(mode!=='live')return;
    try{const headers={};if(lastETag)headers['If-None-Match']=lastETag;
      const r=await fetch('latest.jpg',{headers});if(r.status===304)return;if(!r.ok)return;
      const etag=r.headers.get('ETag');if(etag&&etag===lastETag)return;lastETag=etag;
      const blob=await r.blob();showImage(URL.createObjectURL(blob));
      if(totalFrames>0){slider.value=totalFrames-1;frameLbl.textContent=`${totalFrames} / ${totalFrames}`;}
    }catch(_){}
  }

  setInterval(pollStatus,800);
  setInterval(pollLive,100);

  document.addEventListener('keydown',(e)=>{
    if(e.key==='ArrowLeft')btnPrev.click();if(e.key==='ArrowRight')btnNext.click();
    if(e.key===' '){e.preventDefault();btnPlay.click();}
  });
})();
</script>
</body>
</html>
"""


# ============================================================
# WebBroadcaster
# ============================================================
class WebBroadcaster:
    """
    Lightweight web visualization broadcaster for Genark scripts.

    Usage:
        broadcaster = WebBroadcaster(output_dir="web_output", port=9998)
        for step in range(N):
            rgb = cam.render(...)
            broadcaster.send_frame(rgb, env_idx=0, metadata={"step": step})
        broadcaster.finish()
        broadcaster.wait_for_exit()
    """

    def __init__(self, output_dir="genark_web_output", port=9998, quality=85, throttle_fps=30):
        self._output_dir = os.path.abspath(output_dir)
        self._frames_dir = os.path.join(self._output_dir, "frames")
        self._status_path = os.path.join(self._output_dir, "status.json")
        self._latest_path = os.path.join(self._output_dir, "latest.jpg")
        self._quality = quality
        self._frame_idx = 0
        self._fps = 0.0
        self._running = True
        self._min_interval = 1.0 / throttle_fps if throttle_fps > 0 else 0
        self._last_save_time = 0.0
        self._port = port

        os.makedirs(self._frames_dir, exist_ok=True)

        html_path = os.path.join(self._output_dir, "index.html")
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(_INDEX_HTML)

        self._write_status({})

        _kill_port(port)

        self._server_thread = threading.Thread(
            target=self._serve,
            kwargs={"addr": "0.0.0.0", "port": port, "directory": self._output_dir},
            daemon=True,
        )
        self._server_thread.start()

        print(f"[WebViewer] 浏览器: http://0.0.0.0:{port}/")
        print(f"[WebViewer] 输出目录: {self._output_dir}")

    def _serve(self, addr, port, directory):
        def factory(*args, **kwargs):
            return _ETagHandler(*args, directory=directory, **kwargs)

        try:
            httpd = HTTPServer((addr, port), factory)
            httpd.serve_forever()
        except OSError as e:
            print(f"[WebViewer] HTTP 服务启动失败: {e}")

    def _write_status(self, metadata):
        status = {
            "frame": self._frame_idx,
            "total": -1,
            "running": self._running,
            "fps": self._fps,
            "metadata": metadata,
        }
        try:
            with open(self._status_path, "w") as f:
                json.dump(status, f, ensure_ascii=False)
        except Exception:
            pass

    def send_frame(self, rgb, env_idx=0, metadata=None):
        """
        Broadcast a rendered frame.

        Parameters
        ----------
        rgb : numpy.ndarray or torch.Tensor
            Image data. Can be:
            - (H, W, 3) single frame
            - (N, H, W, 3) batched frames (use env_idx to select)
        env_idx : int
            Which environment to visualize (for batched rendering).
        metadata : dict, optional
            Arbitrary metadata shown in the info panel (episode_id, step, action, etc.).
        """
        now = time.time()
        if self._min_interval > 0 and (now - self._last_save_time) < self._min_interval:
            return
        self._last_save_time = now

        t0 = now

        if hasattr(rgb, "cpu"):
            if rgb.dim() == 4:
                img = rgb[env_idx].cpu().numpy()
            else:
                img = rgb.cpu().numpy()
        else:
            img = np.asarray(rgb)
            if img.ndim == 4:
                img = img[env_idx]

        img = img.astype(np.uint8)

        i = self._frame_idx
        frame_path = os.path.join(self._frames_dir, f"frame_{i:06d}.jpg")
        _save_image(img, frame_path, self._quality)
        _save_image(img, self._latest_path, self._quality)

        dt = time.time() - t0
        self._frame_idx = i + 1
        self._fps = round(1.0 / max(dt, 1e-6), 1)

        self._write_status(metadata or {})

        if (i + 1) % 100 == 0:
            print(f"[WebViewer] 帧 {i + 1}  保存FPS={self._fps}")

    def finish(self):
        """Mark simulation as finished."""
        self._running = False
        total = self._frame_idx
        status = {
            "frame": total,
            "total": total,
            "running": False,
            "fps": 0,
            "metadata": {},
        }
        try:
            with open(self._status_path, "w") as f:
                json.dump(status, f, ensure_ascii=False)
        except Exception:
            pass
        print(f"[WebViewer] 完成, 共 {total} 帧")

    def wait_for_exit(self):
        """Block until Ctrl+C, keeping HTTP server alive for playback."""
        print(f"[WebViewer] 浏览器回放: http://0.0.0.0:{self._port}/")
        print("按 Ctrl+C 退出")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[WebViewer] 退出")
