#include <cstdio>                                              // für printf
#include <cstdint>                                             // für feste Ganzzahltypen (int64_t usw.)
#include "esp_timer.h"                                         // Mikrosekunden-Timer des ESP32
#include "tensorflow/lite/micro/micro_interpreter.h"           // der TFLite-Micro-Interpreter
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"   // Liste der erlaubten Operationen
#include "tensorflow/lite/schema/schema_generated.h"           // liest das .tflite-Format
#include "model_data.h"                                        // unsere beiden Modelle als C-Arrays
#include "test_data.h"                                         // 100 Testbilder + Labels

constexpr int kArenaSize = 80 * 1024;                          // Größe der Tensor-Arena: 80 KB (großzügig, echter Bedarf wird gemessen)
alignas(16) static uint8_t tensor_arena[kArenaSize];           // reserviert den Arbeitsspeicher für Zwischenergebnisse
static tflite::MicroMutableOpResolver<8> resolver;   // Platz für 8 statt 5 Operationstypen

static void run_benchmark(const char* name, const unsigned char* model_data, unsigned int model_len) {
    const tflite::Model* model = tflite::GetModel(model_data);          // liest das Modell aus dem Flash (ohne Kopie)
    auto* interpreter = new tflite::MicroInterpreter(model, resolver, tensor_arena, kArenaSize);  // erstellt den Interpreter
    if (interpreter->AllocateTensors() != kTfLiteOk) {                  // verteilt die Tensoren in der Arena
        printf("[%s] AllocateTensors failed\n", name);                  // Fehlermeldung, falls Arena zu klein
        delete interpreter;                                             // räumt auf
        return;                                                         // bricht ab
    }
    TfLiteTensor* input = interpreter->input(0);                        // Zeiger auf den Eingangstensor
    TfLiteTensor* output = interpreter->output(0);                      // Zeiger auf den Ausgangstensor
    bool is_int8 = (input->type == kTfLiteInt8);                        // erkennt, ob es das int8-Modell ist

    int correct = 0;                                                    // Zähler richtige Vorhersagen
    int64_t total_us = 0, min_us = INT64_MAX, max_us = 0;               // Zeitstatistik in Mikrosekunden

    for (int i = 0; i < NUM_TEST_IMAGES; i++) {                         // geht alle 100 Testbilder durch
        const unsigned char* img = &test_images[i * 784];               // Zeiger auf Bild Nummer i
        for (int j = 0; j < 784; j++) {                                 // geht alle Pixel durch
            if (is_int8) input->data.int8[j] = (int8_t)(img[j] - 128);  // int8: q = Pixel - 128
            else         input->data.f[j] = img[j] / 255.0f;            // float32: Pixel / 255
        }
        int64_t t0 = esp_timer_get_time();                              // Startzeit
        interpreter->Invoke();                                          // führt die Inferenz aus
        int64_t dt = esp_timer_get_time() - t0;                         // gemessene Dauer
        total_us += dt;                                                 // summiert für den Mittelwert
        if (dt < min_us) min_us = dt;                                   // merkt sich die schnellste Inferenz
        if (dt > max_us) max_us = dt;                                   // merkt sich die langsamste Inferenz

        int best = 0;                                                   // Index der bisher besten Klasse
        for (int k = 1; k < 10; k++) {                                  // vergleicht alle 10 Ausgänge
            float a = is_int8 ? output->data.int8[k]    : output->data.f[k];     // aktueller Wert
            float b = is_int8 ? output->data.int8[best] : output->data.f[best];  // bisher bester Wert
            if (a > b) best = k;                                        // neuer Bestwert
        }
        if (best == test_labels[i]) correct++;                          // zählt richtige Vorhersagen
    }

    printf("[%s] model size: %u bytes\n", name, model_len);                               // Modellgröße im Flash
    printf("[%s] arena used: %u bytes\n", name, (unsigned)interpreter->arena_used_bytes()); // tatsächlicher RAM-Bedarf
    printf("[%s] accuracy: %d/%d\n", name, correct, NUM_TEST_IMAGES);                     // Genauigkeit auf dem ESP32
    printf("[%s] latency avg: %.1f us, min: %lld us, max: %lld us\n", name,
           (double)total_us / NUM_TEST_IMAGES, (long long)min_us, (long long)max_us);     // Latenzstatistik
    delete interpreter;                                                 // gibt den Interpreter frei
}

extern "C" void app_main(void) {                       // Einstiegspunkt des Programms
    resolver.AddConv2D();                              // Faltung
    resolver.AddMaxPool2D();                           // Max-Pooling
    resolver.AddReshape();                             // Reshape
    resolver.AddFullyConnected();                      // Dense-Schicht
    resolver.AddSoftmax();                             // Softmax
    resolver.AddShape();                               // NEU: liest die Form eines Tensors aus
    resolver.AddStridedSlice();                        // NEU: schneidet einen Teil davon heraus
    resolver.AddPack();                                // NEU: setzt die neue Form zusammen
    run_benchmark("float32", model_f32, model_f32_len);    // misst das float32-Modell
    run_benchmark("int8", model_int8, model_int8_len);     // misst das int8-Modell
}