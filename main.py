import streamlit as st
import streamlit as st
import serial
import pandas as pd
import time
import os
import threading
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import subprocess
import serial.tools.list_ports
import streamlit as st
import serial
import json
import time
import base64


# Configurar el puerto serial
PORT = "/dev/tty.usbmodem101"
BAUD_RATE = 9600

# Variable global para el estado de grabación
recording = False

# Inicializar estado en session_state para la UI
if "recording" not in st.session_state:
    st.session_state.recording = False
if "movement_type" not in st.session_state:
    st.session_state.movement_type = "Tiro"
if "model_trained" not in st.session_state:
    st.session_state.model_trained = False

def reset_usb_connection(port):
        """Intenta cerrar procesos que usan el puerto serie en macOS."""
        try:
            result = subprocess.run(["lsof", "-t", port], capture_output=True, text=True)
            pids = result.stdout.strip().split("\n")
            for pid in pids:
                if pid:
                    #os.system(f"kill -9 {pid}")  # Forzar cierre del proceso
                    os.system("fuser /dev/tty.usbmodem101 -k")
                    st.write(f"Cerrando proceso {pid} en {port}")
            st.write(f"Reiniciando conexión en {port}...")
        except Exception as e:
            st.error(f"Error al reiniciar la conexión USB: {e}")

def compile_and_upload(arduino_path, board_fqbn, port):
    """Compila y sube el código de Arduino usando arduino-cli."""
    try:
        reset_usb_connection(port)  # Intentar liberar el puerto antes de subir
        
        # Compilar el código
        compile_command = f'arduino-cli compile --fqbn {board_fqbn} "{arduino_path}"'
        st.write(f"Ejecutando: {compile_command}")
        os.system(compile_command)

        # Subir el código
        upload_command = f'arduino-cli upload --fqbn {board_fqbn} --port {port} "{arduino_path}"'
        st.write(f"Ejecutando: {upload_command}")
        os.system(upload_command)

        st.success("Código subido exitosamente al Arduino.")
    except Exception as e:
        st.error(f"Error al subir el código: {e}")

def compile_and_upload_data_collect_code(arduino_path, board_fqbn, port):
    """Compila y sube el código de Arduino usando arduino-cli."""
    try:
        #reset_usb_connection(port)  # Intentar liberar el puerto antes de subir
        
        # Compilar el código
        compile_command = f'arduino-cli compile --fqbn {board_fqbn} "{arduino_path}"'
        st.write(f"Ejecutando: {compile_command}")
        os.system(compile_command)

        # Subir el código
        upload_command = f'arduino-cli upload --fqbn {board_fqbn} --port {port} "{arduino_path}"'
        st.write(f"Ejecutando: {upload_command}")
        os.system(upload_command)

        st.success("Código subido exitosamente al Arduino.")
    except Exception as e:
        st.error(f"Error al subir el código: {e}")


def send_mode_to_arduino(mode, port):
    """Envía el modo de operación al Arduino."""
    try:
        with serial.Serial(port, BAUD_RATE, timeout=1) as ser:
            ser.write(mode.encode())  # Enviar el modo en ASCII
            st.success(f"Modo '{mode}' enviado a Arduino en {port}")
    except Exception as e:
        st.error(f"Error al enviar modo: {e}")

def read_serial(file_name):
    """Lee datos desde el puerto serial del Arduino y los guarda en un archivo CSV."""
    global recording
    try:
        arduino = serial.Serial(PORT, BAUD_RATE, timeout=1)
        time.sleep(2)  # Espera a que Arduino se estabilice
        st.success(f"Conectado a Arduino en {PORT}")
    except Exception as e:
        st.error(f"No se pudo conectar a Arduino: {e}")
        return

    # Si el archivo no existe, escribir la cabecera
    if not os.path.exists(file_name):
        with open(file_name, "w") as f:
            f.write("aX,aY,aZ,gX,gY,gZ\n")

    # Leer datos en tiempo real
    with open(file_name, "a") as f:
        while recording:
            try:
                line = arduino.readline().decode("utf-8").strip()
                if line and "aX" not in line:  # Ignorar la cabecera
                    f.write(line + "\n")
                    print(line)
            except Exception as e:
                print(f"Error al leer datos: {e}")
                break

    arduino.close()
    print("Grabación detenida.")

def read_from_arduino():
    try:
        ser = serial.Serial(PORT, BAUD_RATE, timeout=1)
        while True:
            line = ser.readline().decode("utf-8").strip()
            if line:
                return json.loads(line)
    except Exception as e:
        #pass
        st.error(f"Error en la conexión con Arduino: {e}")
        return None

# INTERFAZ DE USUARIO CON STREAMLIT
st.markdown("<h1 style='text-align: center;'>🏀 The Perfect Game 🏀</h1>", unsafe_allow_html=True)

with st.expander("Programa recolección de datos Arduino"):
    arduino_path = "Capture_Data/Capture_Data.ino"
    board_fqbn = "arduino:mbed_nano:nano33ble"
    
    if st.button("Subir codigo data"):
        if PORT:
            compile_and_upload_data_collect_code(arduino_path, board_fqbn, PORT)
        else:
            st.error("No se ha seleccionado un puerto.")


#--------------------------------------------------------------------------------------------
with st.expander("Entrenamiento del Modelo"):
    st.markdown("---")
    st.header("Captura de datos")
    movement_type = st.selectbox("Selecciona el tipo de movimiento:", ["Tiro", "Pase", "Bote"], index=0)
    st.session_state.movement_type = movement_type
    csv_file = f"{movement_type}.csv"

    # Botón para iniciar la grabación
    if st.button("Iniciar Grabación"):
        if not st.session_state.recording:
            st.session_state.recording = True
            recording = True  # Actualizamos la variable global
            thread = threading.Thread(target=read_serial, args=(csv_file,), daemon=True)
            thread.start()
            st.success(f"Grabación iniciada para {movement_type}...")

    # Botón para detener la grabación
    if st.button("Detener Grabación"):
        if st.session_state.recording:
            st.session_state.recording = False
            recording = False  # Detenemos la grabación
            st.warning("Grabación detenida.")

    # Sección de visualización de datos
    st.markdown("---")
    st.header("Visualizar Datos")
    selected_csv = st.selectbox("Selecciona el archivo CSV a visualizar:", ["Tiro.csv", "Pase.csv", "Bote.csv"])

    if os.path.exists(selected_csv):
        df = pd.read_csv(selected_csv)
        st.dataframe(df)
        
        # Mostrar gráficos
        st.subheader("Gráfica de Datos")
        fig, ax = plt.subplots()
        df.plot(ax=ax)
        st.pyplot(fig)
    else:
        st.warning("No hay datos disponibles para mostrar.")

    # Sección de Entrenamiento del Modelo
    st.markdown("---")
    st.header("Entrenamiento del Modelo")
    if st.button("Entrenar Modelo"):
        st.write("Entrenando modelo...")
        gif_placeholder = st.image("basket_gifs/basket_gif.gif", caption="⏳ Entrenando modelo...")

        SEED = 1337
        np.random.seed(SEED)
        tf.random.set_seed(SEED)

        # the list of gestures that data is available for
        GESTURES = [
            "Bote",
            "Pase",
            "Tiro",
        ]

        SAMPLES_PER_GESTURE = 119

        NUM_GESTURES = len(GESTURES)

        # create a one-hot encoded matrix that is used in the output
        ONE_HOT_ENCODED_GESTURES = np.eye(NUM_GESTURES)

        inputs = []
        outputs = []

        # read each csv file and push an input and output
        for gesture_index in range(NUM_GESTURES):
            gesture = GESTURES[gesture_index]
            print(f"Processing index {gesture_index} for gesture '{gesture}'.")

            output = ONE_HOT_ENCODED_GESTURES[gesture_index]

            df = pd.read_csv(gesture + ".csv")

            # calculate the number of gesture recordings in the file
            num_recordings = int(df.shape[0] / SAMPLES_PER_GESTURE)

            print(f"\tThere are {num_recordings} recordings of the {gesture} gesture.")

            for i in range(num_recordings):
                tensor = []
                for j in range(SAMPLES_PER_GESTURE):
                    index = i * SAMPLES_PER_GESTURE + j
                    # normalize the input data, between 0 to 1:
                    # - acceleration is between: -4 to +4
                    # - gyroscope is between: -2000 to +2000
                    tensor += [
                        (df['aX'][index] + 4) / 8,
                        (df['aY'][index] + 4) / 8,
                        (df['aZ'][index] + 4) / 8,
                        (df['gX'][index] + 2000) / 4000,
                        (df['gY'][index] + 2000) / 4000,
                        (df['gZ'][index] + 2000) / 4000
                    ]
                inputs.append(tensor)
                outputs.append(output)

        # convert the list to numpy array
        inputs = np.array(inputs)
        outputs = np.array(outputs)

        print(f"Nan en inputs: {np.isnan(inputs).sum()}, Nan en outputs:{np.isnan(outputs).sum()}")  # Revisar NaN
        print(f"Nan en inputs: {np.isinf(inputs).sum()}, Nan en outputs:{np.isinf(outputs).sum()}")  # Revisar Inf

        inputs = np.nan_to_num(inputs, nan=0.0, posinf=1.0, neginf=-1.0)
        outputs = np.nan_to_num(outputs, nan=0.0, posinf=1.0, neginf=-1.0)



        print("Data set parsing and preparation complete.")

        num_inputs = len(inputs)
        randomize = np.arange(num_inputs)
        np.random.shuffle(randomize)

        # Swap the consecutive indexes (0, 1, 2, etc) with the randomized indexes
        inputs = inputs[randomize]
        outputs = outputs[randomize]

        # Split the recordings (group of samples) into three sets: training, testing and validation
        TRAIN_SPLIT = int(0.6 * num_inputs)
        TEST_SPLIT = int(0.2 * num_inputs + TRAIN_SPLIT)

        inputs_train, inputs_test, inputs_validate = np.split(inputs, [TRAIN_SPLIT, TEST_SPLIT])
        outputs_train, outputs_test, outputs_validate = np.split(outputs, [TRAIN_SPLIT, TEST_SPLIT])

        print("Data set randomization and splitting complete.")

        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Dense(50, activation='relu')) # relu is used for performance
        model.add(tf.keras.layers.Dense(15, activation='relu'))
        model.add(tf.keras.layers.Dense(NUM_GESTURES, activation='softmax')) # softmax is used, because we only expect one gesture to occur per input
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        history = model.fit(inputs_train, outputs_train, epochs=600, batch_size=1, validation_data=(inputs_validate, outputs_validate))

        loss = history.history['loss']
        val_loss = history.history['val_loss']
        epochs = range(1, len(loss) + 1)

        # Ajuste de la configuración global del tamaño de las figuras
        plt.rcParams["figure.figsize"] = (8, 6)  # Tamaño similar al primero

        # Primer gráfico: Training and Validation Loss
        fig, ax = plt.subplots()
        ax.plot(history.history['loss'], 'g.', label='Training loss')
        ax.plot(history.history['val_loss'], 'b', label='Validation loss')
        ax.set_title('Training and Validation Loss')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        # Segundo gráfico: Training and Validation Loss (Saltando primeros valores)
        SKIP = 100
        fig, ax = plt.subplots()
        ax.plot(epochs[SKIP:], history.history['loss'][SKIP:], 'g.', label='Training loss')
        ax.plot(epochs[SKIP:], history.history['val_loss'][SKIP:], 'b.', label='Validation loss')
        ax.set_title('Training and Validation Loss (Skipped Initial Epochs)')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('Loss')
        ax.legend()
        st.pyplot(fig)

        # Tercer gráfico: Training and Validation MAE
        fig, ax = plt.subplots()
        ax.plot(epochs[SKIP:], history.history['mae'][SKIP:], 'g.', label='Training MAE')
        ax.plot(epochs[SKIP:], history.history['val_mae'][SKIP:], 'b.', label='Validation MAE')
        ax.set_title('Training and Validation MAE')
        ax.set_xlabel('Epochs')
        ax.set_ylabel('MAE')
        ax.legend()
        st.pyplot(fig)
        
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        tflite_model = converter.convert()
        open("gesture_model.tflite", "wb").write(tflite_model)
        
        os.system("xxd -i gesture_model.tflite > program_8/model.h")
        
        st.success("Entrenamiento completado y modelo guardado.")

#--------------------------------------------------------------------------------------------
with st.expander("Guardar el modelo en Arduino"):
    st.markdown("---")
    st.header("Subir Código al Arduino")

    # Configurar paths y placa
    arduino_path = "program_8/program_8.ino"
    board_fqbn = "arduino:mbed_nano:nano33ble"
    
    if st.button("Subir código a Arduino"):
        if PORT:
            compile_and_upload(arduino_path, board_fqbn, PORT)
            st.session_state.model_trained = True
        else:
            st.error("No se ha seleccionado un puerto.")
# #--------------------------------------------------------------------------------------------
if not st.session_state.model_trained:
    st.warning("Para desbloquear la pestaña de predicción de gestos tienes que subir tu modelo al Arduino. ⬆️")
if st.session_state.model_trained:
    gesture_gifs = {
        "tiro": "basket_gifs/tiro_nba.gif",
        "bote": "basket_gifs/bote_nba.gif",
        "pase": "basket_gifs/pase_nba.gif"
    }
    with st.expander("Uso del Modelo"):
        st.subheader("Predicción de Gestos con IMU, Conectado a Arduino Nano 33 BLE")

        # Espacios para mostrar el gesto detectado y el GIF
        gesture_placeholder = st.empty()
        gif_placeholder = st.empty()

        # Bucle de actualización en tiempo real
        while True:
            data = read_from_arduino()
            if data and "gestures" in data:
                predicted_gesture = max(data["gestures"], key=data["gestures"].get)
                gesture_placeholder.markdown(f"## 🖐 Gesto Detectado: **{predicted_gesture}**", unsafe_allow_html=True)

                # Verificar si el archivo existe antes de mostrarlo
                gif_path = gesture_gifs.get(predicted_gesture, None)
                if gif_path and os.path.exists(gif_path):
                    gif_html = f"""
                    <div style="display: flex; justify-content: center;">
                        <img src="data:image/gif;base64,{base64.b64encode(open(gif_path, "rb").read()).decode()}" width="300">
                    </div>
                    """
                    gif_placeholder.markdown(gif_html, unsafe_allow_html=True)
                else:
                    gif_placeholder.empty()  # Limpiar el espacio si no hay GIF disponible

            time.sleep(0.5)  # Para evitar sobrecarga en la CPU