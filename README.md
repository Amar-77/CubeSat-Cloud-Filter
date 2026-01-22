# 🛰️ CubeSat Cloud Filter: Onboard Edge AI for Satellite Bandwidth Optimization

![Project Status](https://img.shields.io/badge/Status-Completed-success?style=for-the-badge&color=2ea44f)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20Lite-Edge%20AI-orange?style=for-the-badge&logo=tensorflow)
![Platform](https://img.shields.io/badge/Platform-Embedded%20%2F%20OBC-lightgrey?style=for-the-badge)

<p align="center">
  <img src="[assets/demo_false_color.png](https://github.com/Amar-77/CubeSat-Cloud-Filter/blob/main/assets/demo_false_color.jpeg?raw=true)" alt="False Color Analysis" width="100%">
  <br>
  <em>Figure 1: Ground Station verification. The system uses False Color Infrared analysis to distinguish between Snow (Cyan) and Clouds (White/Pink), ensuring valuable scientific data is not deleted by mistake.</em>
</p>

---

## 📖 Executive Summary

**70% of Earth is covered by clouds.** For nano-satellites (CubeSats) with limited battery power and expensive downlink bandwidth (approx. **$500/GB**), downloading cloudy, useless images is a critical inefficiency.

**CubeSat-Cloud-Filter** is a flight software simulation that runs a **Quantized Deep Learning Model** on the satellite's Onboard Computer (OBC). It analyzes raw sensor data in real-time (<50ms) and autonomously decides whether to **Downlink** (save) or **Delete** the image.

### 🎯 Key Engineering Goals
1.  **Latency:** Process images in **< 100ms** to keep up with orbital velocity.
2.  **Size:** Model must fit on embedded hardware (< 2MB).
3.  **Accuracy:** Must distinguish **Snow vs. Clouds** (the "White Problem").

---

## 🛠️ Technical Architecture

### 1. The "White Problem" (Physics)
Standard RGB cameras cannot tell the difference between Snow and Clouds—both are bright white.
* **Solution:** We utilize the **Near-Infrared (NIR)** band.
* **Physics:** Clouds are highly reflective in NIR, while Snow absorbs NIR.
* **Implementation:** The AI input tensor consists of 4 channels: `[Red, Green, Blue, NIR]`.

### 2. The Edge AI Brain
The core is a **U-Net** semantic segmentation network, optimized for edge deployment.

| Metric | Specification |
| :--- | :--- |
| **Architecture** | Custom Lightweight U-Net |
| **Input Shape** | `(384, 384, 4)` (Multispectral) |
| **Optimization** | Post-Training Quantization (PTQ) |
| **Precision** | **Int8** (Quantized from Float32) |
| **Model Size** | **1.9 MB** (Reduced from 22 MB) |
| **Inference Time** | **~45 ms** (on standard CPU) |

### 3. Flight Software Pipeline (`flight_software.py`)
This script simulates the Onboard Computer (OBC) loop:
1.  **Sensor Trigger:** Reads 4 separate 16-bit GeoTIFF files from the sensor buffer.
2.  **Preprocessing:** Stacks bands into a single tensor and normalizes values.
3.  **Inference:** Runs the `.tflite` model.
4.  **Logic Gate:**
    * If `Cloud_Cover > 10%` $\rightarrow$ **❌ DELETE** (Save Storage).
    * If `Cloud_Cover < 10%` $\rightarrow$ **✅ DOWNLINK** (Transmit Data).

---

## 🚀 Installation & Usage

### Prerequisites
* Python 3.8+
* `pip` (Python Package Manager)

### 1. Clone the Repository
```bash
git clone [https://github.com/Amar-77/CubeSat-Cloud-Filter.git](https://github.com/Amar-77/CubeSat-Cloud-Filter.git)
cd CubeSat-Cloud-Filter
