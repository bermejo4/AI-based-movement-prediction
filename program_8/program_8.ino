/*
  IMU Classifier - Pase, Tiro, Bote
  Este código usa el acelerómetro y giroscopio del Arduino Nano 33 BLE
  para clasificar movimientos de baloncesto: pase, tiro o bote.

  Basado en el ejemplo de Don Coleman, Sandeep Mistry, Dominic Pajak.
  Modificado para detectar gestos de baloncesto.

  Hardware:
  - Arduino Nano 33 BLE o Arduino Nano 33 BLE Sense.

  Este código está en el dominio público.
*/

#include <Arduino_LSM9DS1.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
#include <tensorflow/lite/version.h>

#include "model.h"  // Asegúrate de reemplazarlo con el nuevo modelo entrenado

const float accelerationThreshold = 2.5; // Umbral de detección de movimiento en G's
const int numSamples = 119; // Número de muestras que toma el modelo

int samplesRead = numSamples;

// Configuración de TensorFlow Lite
tflite::MicroErrorReporter tflErrorReporter;
tflite::AllOpsResolver tflOpsResolver;
const tflite::Model* tflModel = nullptr;
tflite::MicroInterpreter* tflInterpreter = nullptr;
TfLiteTensor* tflInputTensor = nullptr;
TfLiteTensor* tflOutputTensor = nullptr;

// Tamaño del buffer de memoria para TFLite
constexpr int tensorArenaSize = 8 * 1024;
byte tensorArena[tensorArenaSize] __attribute__((aligned(16)));

// Clases de gestos (actualizado)
const char* GESTURES[] = {
  "pase",
  "tiro",
  "bote"
};

#define NUM_GESTURES (sizeof(GESTURES) / sizeof(GESTURES[0]))

void setup() {
  Serial.begin(9600);
  while (!Serial);

  // Inicializar IMU
  if (!IMU.begin()) {
    Serial.println("Error al inicializar el IMU!");
    while (1);
  }

  Serial.print("Tasa de muestreo del acelerómetro: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");

  Serial.print("Tasa de muestreo del giroscopio: ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");

  // Cargar el modelo TFLite
  tflModel = tflite::GetModel(gesture_model_tflite);
  if (tflModel->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Error: versión del modelo incompatible.");
    while (1);
  }

  // Crear el intérprete de TensorFlow Lite
  tflInterpreter = new tflite::MicroInterpreter(tflModel, tflOpsResolver, tensorArena, tensorArenaSize, &tflErrorReporter);
  tflInterpreter->AllocateTensors();

  // Obtener punteros a los tensores de entrada y salida
  tflInputTensor = tflInterpreter->input(0);
  tflOutputTensor = tflInterpreter->output(0);
}

void loop() {
  float aX, aY, aZ, gX, gY, gZ;

  // Esperar a que se detecte un movimiento significativo
  while (samplesRead == numSamples) {
    if (IMU.accelerationAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      float aSum = fabs(aX) + fabs(aY) + fabs(aZ);
      if (aSum >= accelerationThreshold) {
        samplesRead = 0;
        break;
      }
    }
  }

  // Captura de datos para la inferencia
  while (samplesRead < numSamples) {
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      IMU.readAcceleration(aX, aY, aZ);
      IMU.readGyroscope(gX, gY, gZ);

      // Normalización de los datos de entrada
      tflInputTensor->data.f[samplesRead * 6 + 0] = (aX + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 1] = (aY + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 2] = (aZ + 4.0) / 8.0;
      tflInputTensor->data.f[samplesRead * 6 + 3] = (gX + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 4] = (gY + 2000.0) / 4000.0;
      tflInputTensor->data.f[samplesRead * 6 + 5] = (gZ + 2000.0) / 4000.0;

      samplesRead++;

      if (samplesRead == numSamples) {
        // Ejecutar inferencia con el modelo
        if (tflInterpreter->Invoke() != kTfLiteOk) {
          Serial.println("{\"error\": \"Error en la inferencia\"}");
          return;
        }

        // Imprimir los resultados
        Serial.print("{\"gestures\": {");
        for (int i = 0; i < NUM_GESTURES; i++) {
          Serial.print("\"");
          Serial.print(GESTURES[i]);
          Serial.print("\": ");
          Serial.print(tflOutputTensor->data.f[i], 6);
          if (i < NUM_GESTURES - 1) Serial.print(", ");
        }
        Serial.println("}}");
      }
    }
  }
}
