# MiniNN

**Lightweight 8-bit Neural Network Engine for JavaScript**

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.1.0-brightgreen.svg)]()

MiniNN is a zero-dependency, single-file neural network engine that runs in the browser and Node.js.  
Originally powering **Minty** — a tiny chatbot with a real NN backend — now extracted as a standalone open-source library.

---

## Features

- **8-bit weight quantization** — float training internally, int8 view for visualization/storage
- **Multiple activations** — sigmoid, relu, tanh, linear
- **Reinforcement training** — `+1` reinforce, `-1` invert (👍/👎 style)
- **Supervised training** — explicit target vectors
- **JSON Training Data** — write entries in `TrainingData.json`, call one method, weights are saved back automatically
- **Save / load weights** — plain JSON, portable across sessions
- **Dual API** — class-based `new MiniNN()` and functional `MiniNN.create()`
- **Zero dependencies** — one `.js` file, drop it anywhere

---

## Quick Start

### Browser

```html
<script src="mininn.js"></script>
<script>
  const nn = MiniNN.create(6, 8, 8);
  const result = nn.forward([1, 0, 0, 0, 0, 0.5]);
  console.log(result.byte);  // 0–255
</script>
```

### Node.js

```js
const MiniNN = require('./mininn.js');

const nn = new MiniNN(6, 8, 8);
nn.forward([1, 0, 0, 0, 0, 0.5]);
nn.train(1);  // reinforce
```

---

## API

### Creating an instance

```js
// Class-based
const nn = new MiniNN(inputSize, hiddenSize, outputSize, opts);

// Functional
const nn = MiniNN.create(inputSize, hiddenSize, outputSize, opts);
```

**Options:**

| Option | Default | Description |
|---|---|---|
| `lr` | `0.09` | Learning rate |
| `hiddenAct` | `'sigmoid'` | Hidden activation: `sigmoid`, `relu`, `tanh`, `linear` |
| `outputAct` | `'linear'` | Output activation |
| `quantizeHidden` | `true` | Quantize hidden activations to 1/8 steps |

---

### Forward pass

```js
const { hidden, output, bits, byte } = nn.forward([1, 0, 0, 0, 0, 0.5]);
// hidden : hidden layer activations
// output : raw output values
// bits   : binarized output [0,1,1,0,...]
// byte   : bits packed into integer (0–255)
```

---

### Training

```js
// Reinforcement (after a forward pass)
nn.forward(input);
nn.train(1);   // reinforce current output
nn.train(-1);  // invert current output

// Supervised
nn.forward(input);
nn.trainSupervised([0.95, 0.05, 0.95, 0.05, 0.95, 0.05, 0.95, 0.05]);
```

---

### JSON Training Data

The easiest way to train MiniNN is to write a `TrainingData.json` file.  
MiniNN reads it, trains on all entries, then saves the resulting weights back into the same file.

#### TrainingData.json format

```json
{
  "arch": [6, 8, 8],
  "entries": [

    // Reinforcement entry
    { "label": "greeting", "input": [1,0,0,0,0,0.1], "reaction": 1 },
    { "label": "bad response", "input": [1,0,0,0,0,0.1], "reaction": -1 },

    // Supervised entry
    { "label": "explicit target", "input": [0,1,0,0,0,0.4], "target": [0.95,0.05,0.95,0.05,0.95,0.05,0.95,0.05] }

  ],
  "weights": null
}
```

- **`arch`** — must match `[inputSize, hiddenSize, outputSize]` of your MiniNN
- **`entries`** — mixed reinforcement and supervised entries, processed in order
- **`weights`** — `null` on first run; auto-filled by MiniNN after training

#### Node.js — read file, train, write back automatically

```js
const MiniNN = require('./mininn.js');

const nn = new MiniNN(6, 8, 8);

// One call: reads TrainingData.json → trains → saves weights back to the file
const { report } = nn.trainFromFile('./TrainingData.json', { epochs: 10 });

console.log(`Trained ${report.trained} updates, avg loss: ${report.avgLoss.toFixed(4)}`);
```

#### Browser — parse JSON string, train, get updated JSON string back

```js
// e.g. after fetch() or FileReader
const { json, report } = nn.trainFromJSON(rawJsonString, { epochs: 5 });

// json = updated TrainingData.json string with weights embedded
// save it back: localStorage, download link, server POST, etc.
localStorage.setItem('TrainingData', json);

console.log(report.avgLoss);
```

#### Manual step-by-step

```js
const data = JSON.parse(rawJson);         // parse yourself
const report = nn.trainFromData(data, { epochs: 3, shuffle: true });
nn.embedWeights(data);                    // writes weights into data object
const updatedJson = JSON.stringify(data, null, 2);
```

---

### Save / Load weights

```js
// Save
const snapshot = nn.save();              // plain object
const jsonStr  = nn.toJSON();            // JSON string

// Load
nn.load(snapshot);
nn.fromJSON(jsonStr);

// Restore a full instance from saved JSON
const nn2 = MiniNN.fromSaved(jsonStr);
```

---

### Utility

```js
nn.reset();         // reinitialize random weights
nn.toString();      // "MiniNN(6→8→8, lr=0.09)"
nn.inspect();       // { w1, w2, b1, b2 } — current quantized weights
```

---

## TrainingData.json after training

After `trainFromFile()` or `trainFromJSON()`, the `weights` field is populated automatically:

```json
{
  "arch": [6, 8, 8],
  "entries": [ ... ],
  "weights": {
    "version": "1.1.0",
    "arch": [6, 8, 8],
    "lr": 0.09,
    "w1": [[ ... ]],
    "w2": [[ ... ]],
    "b1": [ ... ],
    "b2": [ ... ],
    "_trained_at": "2025-04-12T10:30:00.000Z"
  }
}
```

On the **next run**, MiniNN detects the `weights` field and resumes from there instead of starting fresh.

---

## License

Apache License 2.0 — see [LICENSE](LICENSE)  

