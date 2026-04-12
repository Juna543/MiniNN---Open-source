/*!
 * MiniNN Renderer v1.0.0
 * WebGL + Canvas2D visualization addon for MiniNN
 *
 * Copyright (c) 2025 Juna
 * Licensed under the Apache License, Version 2.0
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Addon for MiniNN (mininn.js) — renders:
 *   - Node graph    : neuron circles + weighted connections (WebGL + Canvas2D overlay)
 *   - Weight heatmap: W1 matrix as a color grid (Canvas2D or DOM)
 *
 * Usage:
 *   const renderer = new MiniNN.Renderer(nn, glCanvas, overlayCanvas, opts);
 *   renderer.start();               // begin render loop
 *   renderer.setActivations(inp, hidden, out);  // feed forward-pass results
 *   renderer.renderHeatmap(container);          // render W1 heatmap to a DOM element
 *   renderer.stop();                // stop render loop
 *
 * Zero dependencies — requires only MiniNN (mininn.js) and a browser with WebGL.
 */

// ─────────────────────────────────────────────────────────────────────────────
//  WEBGL SHADERS
// ─────────────────────────────────────────────────────────────────────────────

const _VS = `
  attribute vec2 aPos;
  uniform vec2  uOff;
  uniform float uScale;
  uniform float uAspect;
  void main() {
    gl_Position  = vec4(aPos.x * uScale / uAspect + uOff.x,
                        aPos.y * uScale            + uOff.y,
                        0.0, 1.0);
    gl_PointSize = uScale * 600.0;
  }
`;

const _FS = `
  precision mediump float;
  uniform vec4 uCol;
  void main() {
    float d = length(gl_PointCoord - 0.5);
    float a = smoothstep(0.5, 0.35, d) + smoothstep(0.5, 0.0, d) * 0.4;
    gl_FragColor = vec4(uCol.rgb, uCol.a * a);
  }
`;

// ─────────────────────────────────────────────────────────────────────────────
//  RENDERER CLASS
// ─────────────────────────────────────────────────────────────────────────────

class MiniNNRenderer {
  /**
   * Create a MiniNN Renderer.
   *
   * @param {MiniNN}          nn            - A MiniNN instance
   * @param {HTMLCanvasElement} glCanvas    - Canvas for WebGL node rendering
   * @param {HTMLCanvasElement} overlayCanvas - Canvas2D for connections + labels (same size, positioned on top)
   * @param {object}          [opts]
   * @param {string[]}        [opts.inputLabels]   - Labels for input nodes  (default: ["I0","I1",...])
   * @param {string[]}        [opts.hiddenLabels]  - Labels for hidden nodes (default: ["H0","H1",...])
   * @param {string[]}        [opts.outputLabels]  - Labels for output nodes (default: ["O0","O1",...])
   * @param {string}          [opts.bgColor]       - WebGL background color hex (default: "#06041a")
   * @param {string}          [opts.activeColor]   - Active node color hex     (default: "#22ffaa")
   * @param {string}          [opts.posEdgeColor]  - Positive edge color rgba  (default: "26,230,130")
   * @param {string}          [opts.negEdgeColor]  - Negative edge color rgba  (default: "200,50,80")
   * @param {number}          [opts.nodeScale]     - Node size scale factor    (default: 0.022)
   * @param {string}          [opts.font]          - Canvas2D label font       (default: "10px monospace")
   */
  constructor(nn, glCanvas, overlayCanvas, opts = {}) {
    this.nn      = nn;
    this.glCv    = glCanvas;
    this.ovCv    = overlayCanvas;
    this._rAF    = null;
    this._aT     = 0;
    this._running = false;

    // Activation state
    this._vizIn  = new Array(nn.iSize).fill(0);
    this._vizH   = new Array(nn.hSize).fill(0);
    this._vizOut = new Array(nn.oSize).fill(0);

    // Options
    this._iLabels = opts.inputLabels  || Array.from({ length: nn.iSize  }, (_, i) => `I${i}`);
    this._hLabels = opts.hiddenLabels || Array.from({ length: nn.hSize  }, (_, i) => `H${i}`);
    this._oLabels = opts.outputLabels || Array.from({ length: nn.oSize  }, (_, i) => `O${i}`);
    this._bg      = opts.bgColor      || '#06041a';
    this._actClr  = opts.activeColor  || '34,255,170';
    this._posEdge = opts.posEdgeColor || '26,230,130';
    this._negEdge = opts.negEdgeColor || '200,50,80';
    this._NS      = opts.nodeScale    || 0.022;
    this._font    = opts.font         || '10px "Space Mono",monospace';

    // Node X positions (3 layers: -0.62, 0, +0.62)
    this._LX = [-0.62, 0.0, 0.62];

    // Precompute node positions in NDC space
    this._pIn  = this._layerPos(nn.iSize,  this._LX[0], 0.72);
    this._pH   = this._layerPos(nn.hSize,  this._LX[1], 0.82);
    this._pOut = this._layerPos(nn.oSize,  this._LX[2], 0.82);

    // Init WebGL
    this.gl = null;
    this._prog = null;
    this._ptBuf = null;
    this._initGL();

    // Init Canvas2D overlay
    this.ctx2 = overlayCanvas ? overlayCanvas.getContext('2d') : null;

    // Auto-resize (browser only)
    this._resizeBound = () => this.resize();
    if (typeof window !== 'undefined')
      window.addEventListener('resize', this._resizeBound);
    this.resize();
  }

  // ── Node position helpers ─────────────────────────────────────────────────

  _layerPos(n, x, spreadY) {
    return Array.from({ length: n }, (_, i) => ({
      x,
      y: n > 1 ? (i / (n - 1) * 2 - 1) * spreadY : 0,
    }));
  }

  // NDC → canvas pixel coords
  _ndc(x, y) {
    const w = this.ovCv ? this.ovCv.width  : this.glCv.width;
    const h = this.ovCv ? this.ovCv.height : this.glCv.height;
    return { cx: (x + 1) / 2 * w, cy: (1 - y) / 2 * h };
  }

  // ── WebGL init ────────────────────────────────────────────────────────────

  _initGL() {
    const gl = this.glCv.getContext('webgl', { antialias: true, alpha: false });
    if (!gl) {
      console.warn('MiniNN.Renderer: WebGL not available. Node rendering disabled.');
      return;
    }
    this.gl = gl;
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);

    const mkShader = (type, src) => {
      const s = gl.createShader(type);
      gl.shaderSource(s, src);
      gl.compileShader(s);
      if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
        console.error('MiniNN.Renderer shader error:', gl.getShaderInfoLog(s));
      return s;
    };

    const prog = gl.createProgram();
    gl.attachShader(prog, mkShader(gl.VERTEX_SHADER,   _VS));
    gl.attachShader(prog, mkShader(gl.FRAGMENT_SHADER, _FS));
    gl.linkProgram(prog);
    this._prog = prog;

    // Single-point buffer (reused for every node)
    this._ptBuf = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this._ptBuf);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([0, 0]), gl.STATIC_DRAW);

    // Cache uniform / attrib locations
    this._loc = {
      aPos:    gl.getAttribLocation(prog,  'aPos'),
      uOff:    gl.getUniformLocation(prog, 'uOff'),
      uScale:  gl.getUniformLocation(prog, 'uScale'),
      uAspect: gl.getUniformLocation(prog, 'uAspect'),
      uCol:    gl.getUniformLocation(prog, 'uCol'),
    };
  }

  // ── Resize ────────────────────────────────────────────────────────────────

  resize() {
    const parent = this.glCv.parentElement;
    const w = parent ? parent.clientWidth  || 300 : 300;
    const h = parent ? parent.clientHeight || 300 : 300;

    this.glCv.width  = w; this.glCv.height  = h;
    if (this.gl) this.gl.viewport(0, 0, w, h);
    if (this.ovCv) { this.ovCv.width = w; this.ovCv.height = h; }
  }

  // ── Public: set activations (call after nn.forward()) ────────────────────

  /**
   * Update displayed activations. Call this after nn.forward() to reflect
   * the latest forward pass in the visualization.
   *
   * @param {number[]} inp    - Input activations  (length = nn.iSize)
   * @param {number[]} hidden - Hidden activations (length = nn.hSize)
   * @param {number[]} output - Output activations (length = nn.oSize)
   * @param {object}   [opts]
   * @param {boolean}  [opts.animate=false] - Animate signal propagation (600ms)
   */
  setActivations(inp, hidden, output, opts = {}) {
    this._vizIn = inp.slice();

    if (!opts.animate) {
      this._vizH   = hidden.slice();
      this._vizOut = output.slice();
      return;
    }

    // Animated forward pass (hidden fades in first, output second)
    const t0 = performance.now(), dur = 600;
    const step = (now) => {
      const t = Math.min((now - t0) / dur, 1);
      this._vizH   = hidden.map(v => v * Math.min(t * 2, 1));
      this._vizOut = output.map(v => v * Math.max((t - 0.5) * 2, 0));
      if (t < 1) requestAnimationFrame(step);
      else {
        this._vizH   = hidden.slice();
        this._vizOut = output.slice();
      }
    };
    requestAnimationFrame(step);
  }

  // ── Public: start / stop render loop ─────────────────────────────────────

  /** Start the continuous render loop. */
  start() {
    if (this._running) return;
    this._running = true;
    const loop = () => {
      if (!this._running) return;
      this._drawGL();
      this._drawOverlay();
      this._rAF = requestAnimationFrame(loop);
    };
    this._rAF = requestAnimationFrame(loop);
  }

  /** Stop the render loop. */
  stop() {
    this._running = false;
    if (this._rAF) cancelAnimationFrame(this._rAF);
    this._rAF = null;
  }

  /**
   * Render a single frame (no loop). Useful for static snapshots.
   */
  renderFrame() {
    this._drawGL();
    this._drawOverlay();
  }

  // ── WebGL node draw ───────────────────────────────────────────────────────

  _hexToRgb(hex) {
    const r = parseInt(hex.slice(1,3),16)/255;
    const g = parseInt(hex.slice(3,5),16)/255;
    const b = parseInt(hex.slice(5,7),16)/255;
    return [r,g,b];
  }

  _drawGL() {
    const gl = this.gl;
    if (!gl) return;

    this._aT += 0.016;
    const W = this.glCv.width, H = this.glCv.height;
    const asp = W / H || 1;

    // Background
    const bg = this._hexToRgb(this._bg);
    gl.clearColor(bg[0], bg[1], bg[2], 1);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.useProgram(this._prog);

    gl.bindBuffer(gl.ARRAY_BUFFER, this._ptBuf);
    gl.enableVertexAttribArray(this._loc.aPos);
    gl.vertexAttribPointer(this._loc.aPos, 2, gl.FLOAT, false, 0, 0);
    gl.uniform1f(this._loc.uAspect, asp);

    const acRGB = this._actClr.split(',').map(Number);

    const drawNode = (x, y, act, isInput) => {
      gl.uniform2f(this._loc.uOff, x, y);
      gl.uniform1f(this._loc.uScale, this._NS);
      const pulse = isInput ? 0 : Math.sin(this._aT * 3 + x * 10 + y * 7) * 0.1;
      const br    = 0.2 + act * 0.8 + pulse;
      if (act > 0.5)
        gl.uniform4f(this._loc.uCol,
          acRGB[0]/255 * br * 0.13,
          acRGB[1]/255 * br,
          acRGB[2]/255 * br * 0.65, 0.92);
      else if (act > 0.1)
        gl.uniform4f(this._loc.uCol, 0.05, 0.5*br+0.2, 0.4*br+0.1, 0.72);
      else
        gl.uniform4f(this._loc.uCol, 0.05, 0.15, 0.12, 0.5);
      gl.drawArrays(gl.POINTS, 0, 1);
    };

    this._pIn.forEach((p, i)  => drawNode(p.x, p.y, this._vizIn[i],  true));
    this._pH.forEach((p, i)   => drawNode(p.x, p.y, this._vizH[i],   false));
    this._pOut.forEach((p, i) => drawNode(p.x, p.y, this._vizOut[i], false));
  }

  // ── Canvas2D overlay: connections + labels ────────────────────────────────

  _drawOverlay() {
    const ctx = this.ctx2;
    if (!ctx) return;
    ctx.clearRect(0, 0, this.ovCv.width, this.ovCv.height);

    const nn = this.nn;

    // Input → Hidden edges
    for (let i = 0; i < nn.iSize; i++) {
      for (let j = 0; j < nn.hSize; j++) {
        const w  = nn.w1[j][i];
        const fs = this._vizIn[i] * this._vizH[j];
        const a  = Math.min(0.02 + (w + 128) / 255 * 0.06 + fs * 0.45, 0.6);
        const from = this._ndc(this._pIn[i].x, this._pIn[i].y);
        const to   = this._ndc(this._pH[j].x,  this._pH[j].y);
        this._drawEdge(ctx, from, to, w >= 0, a, fs > 0.3);
      }
    }

    // Hidden → Output edges
    for (let i = 0; i < nn.hSize; i++) {
      for (let j = 0; j < nn.oSize; j++) {
        const w  = nn.w2[j][i];
        const fs = this._vizH[i] * this._vizOut[j];
        const a  = Math.min(0.02 + (w + 128) / 255 * 0.06 + fs * 0.45, 0.6);
        const from = this._ndc(this._pH[i].x,   this._pH[i].y);
        const to   = this._ndc(this._pOut[j].x, this._pOut[j].y);
        this._drawEdge(ctx, from, to, w >= 0, a, fs > 0.3);
      }
    }

    // Layer header labels
    ctx.font = this._font;
    ctx.fillStyle = `rgba(${this._actClr},0.22)`;
    ctx.textAlign = 'center';
    [
      { x: this._LX[0], t: `INPUT (${nn.iSize})`  },
      { x: this._LX[1], t: `HIDDEN (${nn.hSize})` },
      { x: this._LX[2], t: `OUTPUT (${nn.oSize})` },
    ].forEach(l => {
      const p = this._ndc(l.x, -0.93);
      ctx.fillText(l.t, p.cx, p.cy);
    });

    // Node labels — Input (right-aligned)
    ctx.font = `8px "Space Mono",monospace`;
    this._pIn.forEach((pos, i) => {
      const p = this._ndc(pos.x, pos.y);
      ctx.textAlign = 'left';
      ctx.fillStyle = this._vizIn[i] > 0.5
        ? `rgba(${this._actClr},0.85)`
        : `rgba(${this._actClr},0.28)`;
      ctx.fillText(this._iLabels[i], p.cx + 14, p.cy + 3);
    });

    // Hidden labels
    this._pH.forEach((pos, i) => {
      const p = this._ndc(pos.x, pos.y);
      ctx.textAlign = 'center';
      ctx.fillStyle = this._vizH[i] > 0.5
        ? `rgba(${this._actClr},0.75)`
        : `rgba(${this._actClr},0.2)`;
      ctx.fillText(this._hLabels[i], p.cx, p.cy - 13);
    });

    // Output labels
    this._pOut.forEach((pos, i) => {
      const p = this._ndc(pos.x, pos.y);
      ctx.textAlign = 'right';
      const active = this._vizOut[i] > 0.5;
      ctx.fillStyle = active
        ? `rgba(${this._actClr},0.9)`
        : `rgba(${this._actClr},0.25)`;
      ctx.fillText(
        this._oLabels[i] + (active ? '=1' : '=0'),
        p.cx - 14, p.cy + 3
      );
    });
  }

  _drawEdge(ctx, from, to, positive, alpha, thick) {
    const clr = positive ? this._posEdge : this._negEdge;
    ctx.beginPath();
    ctx.moveTo(from.cx, from.cy);
    ctx.lineTo(to.cx,   to.cy);
    ctx.strokeStyle = `rgba(${clr},${alpha.toFixed(3)})`;
    ctx.lineWidth   = thick ? 1.8 : 0.5;
    ctx.stroke();
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  HEATMAP — W1 weight matrix
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Render the W1 weight heatmap into a DOM container element.
   * Clears the container and fills it with colored cells.
   *
   * @param {HTMLElement} container - Any DOM element to render heatmap into
   * @param {object}      [opts]
   * @param {number}      [opts.cellSize=14]   - Cell size in px
   * @param {number}      [opts.cellGap=2]     - Gap between cells in px
   * @param {boolean}     [opts.showTitle=true] - Show "W1 HEATMAP" title
   * @param {boolean}     [opts.showW2=false]   - Also render W2 below W1
   */
  renderHeatmap(container, opts = {}) {
    if (!container) return;
    const cellSize  = opts.cellSize  ?? 14;
    const cellGap   = opts.cellGap   ?? 2;
    const showTitle = opts.showTitle ?? true;
    const showW2    = opts.showW2    ?? false;

    container.innerHTML = '';
    container.style.cssText += `
      font-family: 'Space Mono', monospace;
      font-size: 9px;
      color: #2a6050;
      display: inline-block;
    `;

    const renderMatrix = (matrix, title) => {
      if (showTitle) {
        const t = document.createElement('div');
        t.textContent = title;
        t.style.cssText = 'color:#22ffaa44;margin-bottom:6px;letter-spacing:0.1em;';
        container.appendChild(t);
      }
      matrix.forEach(row => {
        const rowEl = document.createElement('div');
        rowEl.style.cssText = `display:flex;gap:${cellGap}px;margin-bottom:${cellGap}px;`;
        row.forEach(w => {
          const cell = document.createElement('div');
          cell.title = `w=${w}`;
          cell.style.cssText = `
            width:${cellSize}px;height:${cellSize}px;
            border-radius:2px;
            background:${this._weightColor(w)};
          `;
          rowEl.appendChild(cell);
        });
        container.appendChild(rowEl);
      });
    };

    renderMatrix(this.nn.w1, 'W1 HEATMAP');
    if (showW2) renderMatrix(this.nn.w2, 'W2 HEATMAP');
  }

  /**
   * Render W1 heatmap onto a Canvas2D context (no DOM needed).
   *
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} x       - Top-left X
   * @param {number} y       - Top-left Y
   * @param {number} cellSize
   * @param {number} cellGap
   */
  renderHeatmapCanvas(ctx, x = 0, y = 0, cellSize = 12, cellGap = 2) {
    this.nn.w1.forEach((row, ri) => {
      row.forEach((w, ci) => {
        ctx.fillStyle = this._weightColor(w);
        ctx.fillRect(
          x + ci * (cellSize + cellGap),
          y + ri * (cellSize + cellGap),
          cellSize, cellSize
        );
      });
    });
  }

  _weightColor(w) {
    // w is 8-bit signed: -128 … 127
    const n = (w + 128) / 255;
    return w >= 0
      ? `rgba(20,${Math.round(100 + n * 155)},${Math.round(60 + n * 70)},0.9)`
      : `rgba(${Math.round(100 + (1 - n) * 100)},25,55,0.9)`;
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  DESTROY
  // ─────────────────────────────────────────────────────────────────────────

  /** Clean up WebGL resources and remove event listeners. */
  destroy() {
    this.stop();
    if (typeof window !== 'undefined')
      window.removeEventListener('resize', this._resizeBound);
    const gl = this.gl;
    if (gl) {
      gl.deleteProgram(this._prog);
      gl.deleteBuffer(this._ptBuf);
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  REGISTER AS MiniNN.Renderer ADDON
// ─────────────────────────────────────────────────────────────────────────────

if (typeof MiniNN !== 'undefined') {
  MiniNN.Renderer = MiniNNRenderer;
}

if (typeof module !== 'undefined' && module.exports) {
  module.exports = MiniNNRenderer;
} else if (typeof window !== 'undefined') {
  window.MiniNNRenderer = MiniNNRenderer;
}
