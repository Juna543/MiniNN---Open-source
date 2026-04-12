/*!
 * MiniNN v1.1.0
 * Lightweight 8-bit Neural Network Engine
 *
 * Copyright (c) 2025 Juna
 * Licensed under the Apache License, Version 2.0
 * https://www.apache.org/licenses/LICENSE-2.0
 *
 * Originally powering Minty — a tiny chatbot NN.
 * Extracted and generalized as a standalone open-source engine.
 *
 * Features:
 *   - Feedforward NN with configurable layers
 *   - 8-bit weight quantization (float training, int8 view)
 *   - Multiple activations: sigmoid, relu, tanh, linear
 *   - Backpropagation with reinforcement-style training (+1 / -1)
 *   - Supervised training with explicit targets
 *   - JSON Training Data: load TrainingData.json → auto-train → save weights back
 *   - Save / load weights as plain JSON
 *   - Dual API: class-based & functional
 */

// ─────────────────────────────────────────────────────────────────────────────
//  ACTIVATION FUNCTIONS
// ─────────────────────────────────────────────────────────────────────────────

const Activations = {
  sigmoid: {
    fn:  x => 1 / (1 + Math.exp(-x)),
    d:   s => s * (1 - s),   // derivative given post-activation value
  },
  relu: {
    fn:  x => Math.max(0, x),
    d:   s => (s > 0 ? 1 : 0),
  },
  tanh: {
    fn:  x => Math.tanh(x),
    d:   s => 1 - s * s,
  },
  linear: {
    fn:  x => x,
    d:   _s => 1,
  },
};

// ─────────────────────────────────────────────────────────────────────────────
//  CORE CLASS
// ─────────────────────────────────────────────────────────────────────────────

class MiniNN {
  /**
   * Create a MiniNN instance.
   *
   * @param {number}  inputSize   - Number of input neurons
   * @param {number}  hiddenSize  - Number of hidden neurons
   * @param {number}  outputSize  - Number of output neurons
   * @param {object}  [opts]
   * @param {number}  [opts.lr=0.09]               - Learning rate
   * @param {string}  [opts.hiddenAct='sigmoid']   - Activation for hidden layer
   * @param {string}  [opts.outputAct='linear']    - Activation for output layer
   * @param {boolean} [opts.quantizeHidden=true]   - Quantize hidden activations to 1/8 steps
   */
  constructor(inputSize, hiddenSize, outputSize, opts = {}) {
    this.iSize  = inputSize;
    this.hSize  = hiddenSize;
    this.oSize  = outputSize;
    this.lr     = opts.lr ?? 0.09;

    this._hAct  = Activations[opts.hiddenAct  ?? 'sigmoid'];
    this._oAct  = Activations[opts.outputAct  ?? 'linear'];
    this._quantHidden = opts.quantizeHidden ?? true;

    this._initWeights();

    // Forward-pass cache
    this._inp  = new Array(inputSize).fill(0);
    this._hid  = new Array(hiddenSize).fill(0);
    this._rawH = new Array(hiddenSize).fill(0);
    this._rawO = new Array(outputSize).fill(0);
  }

  // ── Weight init (Xavier uniform) ─────────────────────────────────────────

  _initWeights() {
    const rnd = s => (Math.random() * 2 - 1) * s;
    const s1  = Math.sqrt(6 / (this.iSize + this.hSize));
    const s2  = Math.sqrt(6 / (this.hSize + this.oSize));

    // Float weights — used for all math
    this.w1f = Array.from({ length: this.hSize }, () =>
      Array.from({ length: this.iSize }, () => rnd(s1)));
    this.w2f = Array.from({ length: this.oSize }, () =>
      Array.from({ length: this.hSize }, () => rnd(s2)));
    this.b1  = Array.from({ length: this.hSize }, () => rnd(0.2));
    this.b2  = Array.from({ length: this.oSize }, () => rnd(0.2));

    // 8-bit quantized view — for inspection / visualization
    this.w1 = this._quant(this.w1f);
    this.w2 = this._quant(this.w2f);
  }

  // ── 8-bit quantization  [-128 … 127] ─────────────────────────────────────

  /**
   * Quantize float weights to signed 8-bit integers (scale × 128).
   * @param {number[][]} wf - Float weight matrix
   * @returns {number[][]} Quantized weight matrix
   */
  _quant(wf) {
    return wf.map(row =>
      row.map(v => Math.max(-128, Math.min(127, Math.round(v * 128))))
    );
  }

  /**
   * Dequantize 8-bit weights back to floats.
   * @param {number[][]} wq - Quantized weight matrix
   * @returns {number[][]} Float weight matrix
   */
  _dequant(wq) {
    return wq.map(row => row.map(v => v / 128));
  }

  // ── Hidden activation (optionally quantized to 1/8 steps) ────────────────

  _actH(x) {
    const s = this._hAct.fn(x);
    return this._quantHidden ? Math.round(s * 8) / 8 : s;
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  FORWARD PASS
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Run a forward pass.
   *
   * @param {number[]} inp - Input vector (length = inputSize)
   * @returns {{ hidden: number[], output: number[], bits: number[], byte: number }}
   *   - hidden : hidden activations
   *   - output : raw output activations
   *   - bits   : binarized output (threshold = 0)
   *   - byte   : output bits packed into a single integer (up to 8 bits)
   */
  forward(inp) {
    this._inp = inp.slice();

    // Hidden layer
    this._hid = this.w1f.map((row, i) => {
      let sum = this.b1[i];
      for (let j = 0; j < this.iSize; j++) sum += row[j] * (inp[j] ?? 0);
      this._rawH[i] = sum;
      return this._actH(sum);
    });

    // Output layer
    this._rawO = this.w2f.map((row, i) => {
      let sum = this.b2[i];
      for (let j = 0; j < this.hSize; j++) sum += row[j] * this._hid[j];
      this._rawO[i] = sum;
      return this._oAct.fn(sum);
    });

    const bits = this._rawO.map(v => (v >= 0 ? 1 : 0));
    const byte = bits.slice(0, 8).reduce((acc, b, i) => acc | (b << (7 - i)), 0);

    return {
      hidden: this._hid.slice(),
      output: this._rawO.slice(),
      bits,
      byte,
    };
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  TRAINING  (reinforcement-style: +1 reinforce, -1 invert)
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Train with a thumbs-up / thumbs-down signal.
   *
   * @param {1 | -1} reaction  +1 = reinforce current output, -1 = invert it
   * @returns {number} MSE loss after update
   */
  train(reaction) {
    // Build targets
    const target = this._rawO.map(v => {
      const cur = v >= 0 ? 1.0 : 0.0;
      return reaction > 0
        ? cur                           // reinforce
        : (cur > 0.5 ? 0.05 : 0.95);   // invert
    });

    // Output delta  (use sigmoid regardless of outputAct for stable gradients)
    const sig  = x => 1 / (1 + Math.exp(-x));
    const sigD = s => s * (1 - s);

    const rawO = this.w2f.map((row, i) => {
      let sum = this.b2[i];
      for (let j = 0; j < this.hSize; j++) sum += row[j] * this._hid[j];
      return sum;
    });

    const oErr = rawO.map((raw, i) => {
      const a = sig(raw);
      return (target[i] - a) * sigD(a);
    });

    // Hidden delta
    const hErr = this._hid.map((h, j) => {
      let e = 0;
      for (let i = 0; i < this.oSize; i++) e += oErr[i] * this.w2f[i][j];
      return e * this._hAct.d(h);
    });

    // Weight updates
    for (let i = 0; i < this.oSize; i++) {
      for (let j = 0; j < this.hSize; j++)
        this.w2f[i][j] += this.lr * oErr[i] * this._hid[j];
      this.b2[i] += this.lr * oErr[i];
    }
    for (let j = 0; j < this.hSize; j++) {
      for (let k = 0; k < this.iSize; k++)
        this.w1f[j][k] += this.lr * hErr[j] * this._inp[k];
      this.b1[j] += this.lr * hErr[j];
    }

    // Refresh 8-bit quantized view
    this.w1 = this._quant(this.w1f);
    this.w2 = this._quant(this.w2f);

    return oErr.reduce((s, e) => s + e * e, 0) / this.oSize; // MSE
  }

  /**
   * Train with explicit float targets (supervised).
   *
   * @param {number[]} targets - Target values, length = outputSize
   * @returns {number} MSE loss
   */
  trainSupervised(targets) {
    const sig  = x => 1 / (1 + Math.exp(-x));
    const sigD = s => s * (1 - s);

    const rawO = this.w2f.map((row, i) => {
      let sum = this.b2[i];
      for (let j = 0; j < this.hSize; j++) sum += row[j] * this._hid[j];
      return sum;
    });

    const oErr = rawO.map((raw, i) => {
      const a = sig(raw);
      return ((targets[i] ?? 0) - a) * sigD(a);
    });

    const hErr = this._hid.map((h, j) => {
      let e = 0;
      for (let i = 0; i < this.oSize; i++) e += oErr[i] * this.w2f[i][j];
      return e * this._hAct.d(h);
    });

    for (let i = 0; i < this.oSize; i++) {
      for (let j = 0; j < this.hSize; j++)
        this.w2f[i][j] += this.lr * oErr[i] * this._hid[j];
      this.b2[i] += this.lr * oErr[i];
    }
    for (let j = 0; j < this.hSize; j++) {
      for (let k = 0; k < this.iSize; k++)
        this.w1f[j][k] += this.lr * hErr[j] * this._inp[k];
      this.b1[j] += this.lr * hErr[j];
    }

    this.w1 = this._quant(this.w1f);
    this.w2 = this._quant(this.w2f);

    return oErr.reduce((s, e) => s + e * e, 0) / this.oSize;
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  SAVE / LOAD
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Serialize weights to a plain JSON-safe object.
   * Stores 8-bit quantized weights (compact) + biases as floats.
   *
   * @returns {object} Serializable weights snapshot
   */
  save() {
    return {
      version:  '1.0.0',
      arch:     [this.iSize, this.hSize, this.oSize],
      lr:       this.lr,
      hiddenAct: this._quantHidden ? 'sigmoid_q' : 'sigmoid',
      w1: this.w1,   // 8-bit quantized
      w2: this.w2,
      b1: this.b1.map(v => +v.toFixed(6)),
      b2: this.b2.map(v => +v.toFixed(6)),
    };
  }

  /**
   * Load weights from a saved snapshot (produced by `.save()`).
   * Dequantizes w1/w2 back to floats for continued training.
   *
   * @param {object} snapshot - Object previously returned by `.save()`
   */
  load(snapshot) {
    if (!snapshot || !snapshot.arch) throw new Error('MiniNN: invalid snapshot');
    const [iS, hS, oS] = snapshot.arch;
    if (iS !== this.iSize || hS !== this.hSize || oS !== this.oSize)
      throw new Error(`MiniNN: arch mismatch. Expected [${this.iSize},${this.hSize},${this.oSize}]`);

    this.lr   = snapshot.lr ?? this.lr;
    this.w1   = snapshot.w1;
    this.w2   = snapshot.w2;
    this.b1   = snapshot.b1;
    this.b2   = snapshot.b2;

    // Restore float weights from 8-bit quantized
    this.w1f  = this._dequant(this.w1);
    this.w2f  = this._dequant(this.w2);
  }

  /**
   * Export weights as a JSON string.
   * @returns {string}
   */
  toJSON() { return JSON.stringify(this.save(), null, 2); }

  /**
   * Load weights from a JSON string.
   * @param {string} json
   */
  fromJSON(json) { this.load(JSON.parse(json)); }

  // ─────────────────────────────────────────────────────────────────────────
  //  JSON TRAINING DATA
  // ─────────────────────────────────────────────────────────────────────────

  /**
   * Train from a TrainingData object (parsed from TrainingData.json).
   *
   * Supported entry formats (mixed allowed):
   *
   *   Reinforcement:
   *   { "input": [0,1,0,1,0,0.5], "reaction": 1 }   // +1 reinforce, -1 invert
   *
   *   Supervised:
   *   { "input": [0,1,0,1,0,0.5], "target": [1,0,1,0,1,0,1,0] }
   *
   * The object may already contain a "weights" field from a previous run —
   * MiniNN will load those weights first before continuing training.
   *
   * @param {object}  data            - Parsed TrainingData.json object
   * @param {object}  [opts]
   * @param {number}  [opts.epochs=1] - How many times to loop over all entries
   * @param {boolean} [opts.shuffle=false] - Shuffle entries each epoch
   * @returns {{ losses: number[], avgLoss: number, trained: number }}
   *   - losses   : loss value per entry × epoch
   *   - avgLoss  : mean loss across all entries
   *   - trained  : total number of weight updates performed
   */
  trainFromData(data, opts = {}) {
    if (!data || !Array.isArray(data.entries))
      throw new Error('MiniNN: TrainingData must have an "entries" array');

    // If the file already has saved weights, resume from them
    if (data.weights) {
      try { this.load(data.weights); } catch (_) { /* arch mismatch → start fresh */ }
    }

    const epochs  = opts.epochs  ?? 1;
    const shuffle = opts.shuffle ?? false;
    const losses  = [];
    let trained   = 0;

    for (let ep = 0; ep < epochs; ep++) {
      let entries = data.entries.slice();
      if (shuffle) entries = entries.sort(() => Math.random() - 0.5);

      for (const entry of entries) {
        if (!Array.isArray(entry.input)) continue;

        this.forward(entry.input);

        let loss;
        if (Array.isArray(entry.target)) {
          // Supervised
          loss = this.trainSupervised(entry.target);
        } else if (entry.reaction === 1 || entry.reaction === -1) {
          // Reinforcement
          loss = this.train(entry.reaction);
        } else {
          continue; // skip malformed entry
        }

        losses.push(loss);
        trained++;
      }
    }

    const avgLoss = losses.length
      ? losses.reduce((s, v) => s + v, 0) / losses.length
      : 0;

    return { losses, avgLoss, trained };
  }

  /**
   * Save current weights back into a TrainingData object so it can be
   * written to TrainingData.json (entries are preserved, weights field updated).
   *
   * @param {object} data - The same TrainingData object passed to trainFromData()
   * @returns {object} Updated data object (mutated in place + returned)
   */
  embedWeights(data) {
    data.weights        = this.save();
    data.weights._trained_at = new Date().toISOString();
    return data;
  }

  /**
   * Full pipeline (browser): parse JSON string → train → return updated JSON string.
   * Write the result back to your file / localStorage / wherever.
   *
   * @param {string}  json            - Raw contents of TrainingData.json
   * @param {object}  [opts]
   * @param {number}  [opts.epochs=1]
   * @param {boolean} [opts.shuffle=false]
   * @returns {{ json: string, report: object }}
   *   - json   : updated TrainingData.json string (weights embedded)
   *   - report : { losses, avgLoss, trained }
   */
  trainFromJSON(json, opts = {}) {
    const data   = JSON.parse(json);
    const report = this.trainFromData(data, opts);
    this.embedWeights(data);
    return { json: JSON.stringify(data, null, 2), report };
  }

  /**
   * Full pipeline (Node.js): read TrainingData.json → train → write weights back.
   *
   * Requires Node.js `fs` module. Safe to call in browser too — it will
   * throw a clear error if `fs` is unavailable.
   *
   * @param {string}  filePath        - Path to TrainingData.json
   * @param {object}  [opts]
   * @param {number}  [opts.epochs=1]
   * @param {boolean} [opts.shuffle=false]
   * @returns {{ report: object }} Training summary
   */
  trainFromFile(filePath, opts = {}) {
    let fs;
    try { fs = require('fs'); } catch (_) {
      throw new Error('MiniNN.trainFromFile() requires Node.js (fs module not available in browser — use trainFromJSON() instead)');
    }
    const raw    = fs.readFileSync(filePath, 'utf8');
    const data   = JSON.parse(raw);
    const report = this.trainFromData(data, opts);
    this.embedWeights(data);
    fs.writeFileSync(filePath, JSON.stringify(data, null, 2), 'utf8');
    return { report };
  }

  // ─────────────────────────────────────────────────────────────────────────
  //  UTILITY
  // ─────────────────────────────────────────────────────────────────────────

  /** Reset to fresh random weights. */
  reset() { this._initWeights(); }

  /** Human-readable architecture string. */
  toString() {
    return `MiniNN(${this.iSize}→${this.hSize}→${this.oSize}, lr=${this.lr})`;
  }

  /**
   * Get a flat summary of current quantized weights for debugging.
   * @returns {{ w1: number[][], w2: number[][], b1: number[], b2: number[] }}
   */
  inspect() {
    return { w1: this.w1, w2: this.w2, b1: this.b1, b2: this.b2 };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  FUNCTIONAL API
// ─────────────────────────────────────────────────────────────────────────────

const MiniNNFactory = {
  /**
   * Create a new MiniNN instance (functional style).
   *
   * @param {number} inputSize
   * @param {number} hiddenSize
   * @param {number} outputSize
   * @param {object} [opts]
   * @returns {MiniNN}
   *
   * @example
   * const nn = MiniNN.create(6, 8, 8);
   */
  create(inputSize, hiddenSize, outputSize, opts = {}) {
    return new MiniNN(inputSize, hiddenSize, outputSize, opts);
  },

  /**
   * Load a MiniNN from a saved JSON string.
   *
   * @param {string} json - JSON string produced by `nn.toJSON()`
   * @returns {MiniNN}
   *
   * @example
   * const nn = MiniNN.fromSaved(localStorage.getItem('weights'));
   */
  fromSaved(json) {
    const snap = JSON.parse(json);
    const [iS, hS, oS] = snap.arch;
    const nn = new MiniNN(iS, hS, oS, { lr: snap.lr });
    nn.load(snap);
    return nn;
  },

  /** Expose activation presets for custom usage. */
  Activations,

  /** Expose class for instanceof checks. */
  MiniNN,
};

// ─────────────────────────────────────────────────────────────────────────────
//  EXPORT  (works in browser globals, CommonJS, and ESM via bundler)
// ─────────────────────────────────────────────────────────────────────────────

// Attach functional API methods directly onto the class for convenience:
// MiniNN.create(...)  /  MiniNN.fromSaved(...)  /  MiniNN.Activations
Object.assign(MiniNN, MiniNNFactory);

if (typeof module !== 'undefined' && module.exports) {
  // CommonJS (Node.js)
  module.exports = MiniNN;
} else if (typeof window !== 'undefined') {
  // Browser global
  window.MiniNN = MiniNN;
}
