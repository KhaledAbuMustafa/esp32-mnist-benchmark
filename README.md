# ESP32 MNIST Benchmark: float32 vs. int8 with TensorFlow Lite Micro

Running a small CNN on a classic ESP32 and measuring how it behaves under embedded constraints: **latency, memory (flash/RAM) and accuracy**, comparing a float32 model with its full-integer int8 quantized version.

## Goal

- Train a small neural network on a PC/cloud and deploy it to a microcontroller.
- Apply post-training int8 quantization and quantify the trade-offs on real hardware.
- Investigate how hardware and build settings (compiler optimization, CPU clock) affect inference time.

## Setup

| Item | Details |
|---|---|
| Board | ESP32 DevKit V1 (ESP32, dual-core Xtensa LX6, silicon rev. v3.1, 4 MB flash, no PSRAM used) |
| Framework | ESP-IDF v6.1 |
| Inference engine | [esp-tflite-micro](https://components.espressif.com/components/espressif/esp-tflite-micro) (version pinned in `dependencies.lock`) |
| Training | TensorFlow / Keras in Google Colab |
| Dataset | MNIST (28×28 grayscale digits) |

### Model

```
Input 28x28x1
Conv2D 8 filters 3x3, ReLU   -> 26x26x8
MaxPooling 2x2               -> 13x13x8
Conv2D 16 filters 3x3, ReLU  -> 11x11x16
MaxPooling 2x2               -> 5x5x16
Flatten                      -> 400
Dense 10, Softmax
```

5,258 parameters. Trained for 5 epochs (Adam, batch size 64, 10 % validation split).

### Quantization

- **float32:** converted to `.tflite` without quantization.
- **int8:** full-integer post-training quantization with a representative dataset of 200 training images for calibration. All operators, input and output are int8.
- Input quantization: scale = 1/255, zero point = −128, so the int8 input is simply `pixel − 128` (no floating-point preprocessing needed on the MCU).

### Measurement method

- Models and 100 test images (first 100 of the MNIST test set) are compiled into flash as C arrays.
- Latency: `esp_timer_get_time()` around `interpreter->Invoke()` only (no preprocessing), averaged over 100 inferences. Min/max are reported; the max includes the first (cold) inference.
- RAM: `arena_used_bytes()` of the TFLite Micro tensor arena after `AllocateTensors()`.
- Full-test-set accuracy (10,000 images) was measured in Python with the TFLite interpreter on the same `.tflite` files.

## Results

### Model comparison (ESP32 @ 240 MHz)

| Metric | float32 | int8 | int8 vs. float32 |
|---|---|---|---|
| Model size (`.tflite`, flash) | 24,992 B | 10,552 B | 2.4× smaller |
| Tensor arena (RAM) | 28,664 B | 8,524 B | 3.4× smaller |
| Latency avg | 58.7 ms | 23.7 ms | 2.5× faster |
| Latency min / max | 58.3 / 91.5 ms | 23.6 / 24.9 ms | |
| Accuracy, Python (10,000 images) | 98.20 % | 98.16 % | −0.04 pp |
| Accuracy, on ESP32 (100 images) | 99/100 | 99/100 | |

### Effect of build settings on average latency

| Configuration | float32 | int8 |
|---|---|---|
| 160 MHz, `-Og` (default debug build) | 87.0 ms | 35.5 ms |
| 160 MHz, `-O2` | 87.0 ms | 35.5 ms |
| 240 MHz, `-O2` | 58.7 ms | 23.7 ms |

### Firmware footprint (`idf.py size`, whole benchmark firmware)

| Section | Used |
|---|---|
| Total image size | 339,697 B |
| Flash data (`.rodata`, incl. both models and 100 test images) | 162,736 B |
| Flash code | 125,284 B |
| DRAM | 95,518 B of 180,736 B (incl. 80 KB statically reserved tensor arena) |
| IRAM | 40,975 B of 131,072 B |

## Findings

1. **int8 pays off on every metric that matters on an MCU.** It is 2.5× faster, needs 3.4× less RAM and costs only 0.04 percentage points of accuracy.
2. **Model size shrinks less than the expected 4×.** The weights shrink by 4×, but for such a small network the fixed `.tflite` overhead (graph structure, metadata, quantization parameters) and the int32 biases make up a large share of the file.
3. **Project-wide compiler optimization had no measurable effect.** Inspecting the component's `CMakeLists.txt` showed that esp-tflite-micro builds its kernels with its own `-O3` flag, so `-Og` vs. `-O2` only affected the application code, not the inference kernels.
4. **Inference is compute-bound.** Raising the CPU clock from 160 to 240 MHz gave a speedup of 1.48× (float32) and 1.50× (int8), almost exactly the clock ratio of 1.5.
5. **Keras `Flatten` is not a single operator after conversion.** It becomes `SHAPE`, `STRIDED_SLICE`, `PACK` and `RESHAPE`, which all had to be registered in the `MicroMutableOpResolver`. The first deployment failed with `Didn't find op for builtin opcode 'SHAPE'`.
6. **The tensor arena is oversized.** 80 KB are reserved statically, but the measured peak is 28.7 KB (float32) and 8.5 KB (int8). For an int8-only deployment about 10 KB would be sufficient, which matters on smaller MCUs.

## Limitations

- **Energy was not measured** (no suitable measurement equipment available).
- On-device accuracy is based on only 100 images and cannot resolve the 0.04 pp difference. It confirms that on-device inference is correct; the Python results are the reference for accuracy.
- The 100 on-device test images are the first 100 of the MNIST test set, not a random sample.
- The classic ESP32 does not get the assembly-optimized ESP-NN kernels that are available for the ESP32-S3, so these latencies are not representative of newer ESP32 variants.
- `tf.lite.Interpreter` is deprecated in favor of LiteRT (`ai_edge_litert`); the Python evaluation still used it.

## Repository structure

```
├── CMakeLists.txt              # ESP-IDF project file
├── main/
│   ├── main.cc                 # benchmark: runs both models, measures latency/RAM/accuracy
│   ├── model_data.h            # float32 and int8 models as C arrays
│   ├── test_data.h             # 100 MNIST test images + labels
│   ├── CMakeLists.txt
│   └── idf_component.yml       # esp-tflite-micro dependency
├── training/
│   └── train_and_convert.ipynb # training, quantization, Python evaluation, C array export
├── sdkconfig                   # 240 MHz, -O2, 4 MB flash
└── dependencies.lock           # pinned component versions
```

## How to reproduce

1. Run `training/train_and_convert.ipynb` (e.g. in Google Colab) to train, convert and export `model_data.h` and `test_data.h` into `main/`.
2. Install ESP-IDF v6.1.
3. Build, flash and monitor:
   ```bash
   idf.py set-target esp32
   idf.py -p <PORT> build flash monitor
   ```

## What I learned

- The full workflow from Keras training to TFLite conversion to inference with TFLite Micro on an ESP32 using ESP-IDF.
- Applying full-integer post-training quantization in practice and handling int8 input quantization on the device, building on the quantization concepts (scale factors, zero points, calibration) from my university course on hardware for AI.
- Measuring before optimizing: a build setting that "should" speed things up had no effect, and finding the reason required looking into the build configuration of the inference library.
- Debugging an embedded ML toolchain: CMake project structure, component manager dependencies and missing operator registrations.
