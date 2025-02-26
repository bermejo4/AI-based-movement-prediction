# The Perfect Game: AI-based Movement Prediction

This repository contains the necessary code to develop an AI-based movement prediction model using Arduino and Streamlit. The project captures motion data from an Arduino board, processes it, and trains a machine learning model to classify different types of basketball movements: **Shot (Tiro)**, **Pass (Pase)**, and **Dribble (Bote)**.

## Features
- **Real-time Data Capture**: Collect movement data from an Arduino Nano 33 BLE Sense.
- **Machine Learning Model**: Train a TensorFlow-based model to classify different basketball actions.
- **Streamlit Web Interface**: An interactive dashboard for data collection, visualization, and model training.
- **Arduino Integration**: Automate data collection and model deployment on an Arduino device.

## Project Structure
```
├── basket_gifs/          # GIFs for UI animations
│   ├── basket_gif.gif
│   ├── bote_nba.gif
│   ├── pase_nba.gif
│   ├── tiro_nba.gif
│
├── Capture_Data/         # Arduino sketch for data collection
│   ├── Capture_Data.ino
│
├── program_8/            # Arduino sketch for model deployment
│   ├── model.h           # Model weights for embedded inference
│   ├── program_8.ino
│
├── gesture_model.tflite  # Trained TensorFlow Lite model
├── main.py               # Streamlit application for UI and data processing
├── Bote.csv              # Data for dribbling movement
├── Pase.csv              # Data for passing movement
├── Tiro.csv              # Data for shooting movement
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
```

## Installation
### Prerequisites
- Python 3.8+
- Arduino Nano 33 BLE Sense
- [Arduino CLI](https://arduino.github.io/arduino-cli/)

### Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/AI-based-movement-prediction.git
   cd AI-based-movement-prediction
   ```
2. Install dependencies:
   ```sh
   pip install -r requirements.txt
   ```
3. Upload the data collection code to Arduino:
   - Connect your Arduino device.
   - Run the following command:
     ```sh
     arduino-cli compile --fqbn arduino:mbed_nano:nano33ble Capture_Data.ino
     arduino-cli upload --fqbn arduino:mbed_nano:nano33ble --port /dev/ttyUSB0 Capture_Data.ino
     ```
4. Start the Streamlit dashboard:
   ```sh
   streamlit run main.py
   ```

## How It Works
### 1. Data Collection
- The **Arduino Nano 33 BLE Sense** collects IMU sensor data (accelerometer and gyroscope values) while performing basketball movements.
- The collected data is saved in CSV format (`Tiro.csv`, `Pase.csv`, `Bote.csv`).

### 2. Model Training
- The `main.py` script preprocesses the collected data and trains a neural network model using **TensorFlow**.
- The model is converted to **TensorFlow Lite** for deployment on Arduino.

### 3. Model Deployment
- The trained model (`gesture_model.tflite`) is embedded into `model.h` for deployment on the Arduino board.
- The `program_8.ino` sketch loads the model and performs real-time movement classification.

## Usage
1. **Start Data Collection**
   - Open the Streamlit UI.
   - Select the movement type (**Shot, Pass, Dribble**) and start recording data.
2. **Train the Model**
   - Click "Train Model" in the UI.
   - The model is trained and saved as a TFLite file.
3. **Deploy to Arduino**
   - Upload the `program_8.ino` file to Arduino.
   - The device will classify movements in real-time.

## Dependencies
The following Python packages are required (listed in `requirements.txt`):
```txt
streamlit==1.41.1
pyserial==3.5
pandas==2.2.3
matplotlib==3.10.0
numpy==2.0.2
tensorflow==2.18.0
bleak==0.22.3
```
Install them using:
```sh
pip install -r requirements.txt
```

## Future Improvements
- Improve model accuracy with more data and feature engineering.
- Implement real-time classification feedback on Arduino.
- Enhance the UI with additional visualization options.

## License
This project is licensed under the MIT License.

---
Developed by bermejo4

