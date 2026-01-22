import os
import time
import random
import cv2
import numpy as np
import tensorflow as tf
import shutil
import matplotlib.pyplot as plt

# --- 🛰️ SATELLITE CONFIGURATION ---
RAW_DATA_DIR = os.path.join("Amar_Satellite_Raw_Data", "camera_raw_data")
MODEL_PATH = os.path.join("saved_models", "model_quantized.tflite")
DOWNLINK_DIR = "to_downlink"

CLOUD_THRESHOLD = 0.10  # 10% Cloud Cover limit

# Create Downlink Buffer if it doesn't exist
if os.path.exists(DOWNLINK_DIR):
    shutil.rmtree(DOWNLINK_DIR)
os.makedirs(DOWNLINK_DIR)

# --- 1. INITIALIZE ONBOARD AI ---
print("🛰️ SYSTEM BOOT: Loading Neural Engine...")
try:
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("✅ AI ONLINE. Ready for sensor data.\n")
except Exception as e:
    print(f"❌ SYSTEM FAILURE: Model not found at {MODEL_PATH}")
    print(f"   Error: {e}")
    exit()

def simulate_camera_capture():
    """
    Simulates the camera sensor triggering.
    It randomly picks a 'scene' (file prefix) from the raw data folder
    and reads the 4 separate bands into RAM.
    """
    # 1. Randomly decide to point at a Cloudy or Clear area
    category = random.choice(['clear','cloudy'])
    base_path = os.path.join(RAW_DATA_DIR, category)
    
    # 2. Find a valid image ID (Looking at Blue band list)
    blue_folder = os.path.join(base_path, 'blue')
    if not os.path.exists(blue_folder): return None, None
    
    files = os.listdir(blue_folder)
    if not files: return None, None
    
    target_file = random.choice(files) # e.g., "blue_patch_100.TIF"
    # Extract the ID: "patch_100.TIF"
    img_id = target_file.replace('blue_', '') 
    
    print(f"📸 CAMERA TRIGGERED: Capturing Scene {img_id[:15]}... ({category.upper()})")
    
    # 3. READ RAW SENSOR STREAMS (Simulating separate bands)
    # Note: We reconstruct the paths for Green, Red, NIR
    b_path = os.path.join(base_path, 'blue',  f"blue_{img_id}")
    g_path = os.path.join(base_path, 'green', f"green_{img_id}")
    r_path = os.path.join(base_path, 'red',   f"red_{img_id}")
    n_path = os.path.join(base_path, 'nir',   f"nir_{img_id}")
    
    # Read into RAM
    b = cv2.imread(b_path, cv2.IMREAD_UNCHANGED)
    g = cv2.imread(g_path, cv2.IMREAD_UNCHANGED)
    r = cv2.imread(r_path, cv2.IMREAD_UNCHANGED)
    n = cv2.imread(n_path, cv2.IMREAD_UNCHANGED)
    
    if b is None or n is None:
        print("   ⚠️ SENSOR ERROR: Incomplete Data Stream.")
        return None, None

    return (b, g, r, n), img_id

def process_and_decide(bands, img_id):
    """
    The Onboard Computer (OBC) Logic.
    Stacks RAM data -> Normalizes -> Runs AI -> Makes Decision.
    """
    b, g, r, n = bands
    
    # --- STEP A: BUFFERING (Stacking in RAM) ---
    # Real satellites stack raw bands into a single memory block here
    img_stack = np.dstack((r, g, b, n))
    
    # --- STEP B: PREPROCESSING ---
    # Normalize 16-bit (0-65535) to Float (0-1)
    input_tensor = img_stack.astype(np.float32) / 65535.0
    input_tensor = np.expand_dims(input_tensor, axis=0) # Batch of 1
    
    # --- STEP C: AI INFERENCE ---
    interpreter.set_tensor(input_details[0]['index'], input_tensor)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
   # --- STEP D: DECISION LOGIC ---
    # Calculate Cloud Percentage
    prediction = output_data[0, :, :, 0]
    cloud_pixels = np.sum(prediction > 0.5)
    total_pixels = prediction.size
    cloud_pct = cloud_pixels / total_pixels
    
    print(f"   🧠 AI ANALYSIS: Cloud Cover = {cloud_pct:.1%}")
    
    if cloud_pct > CLOUD_THRESHOLD:
        print(f"   ❌ ACTION: DELETE (Too Cloudy). Save Storage.\n")
        status = "DELETED"
    else:
        print(f"   ✅ ACTION: DOWNLINK (Clear). Writing to Buffer...\n")
        status = "SENT"
        
        # 1. SAVE SCIENTIFIC DATA (The Real Product)
        # Saves 4 Bands (Blue, Green, Red, NIR) + 16-bit Precision
        npy_path = os.path.join(DOWNLINK_DIR, f"packet_{img_id}.npy")
        np.save(npy_path, img_stack)
        
        # 2. SAVE PRETTY PREVIEW (For Human Verification)
        # This applies the "Satellite Instagram Filter" so it doesn't look like trash
        jpg_path = os.path.join(DOWNLINK_DIR, f"view_{img_id}.jpg")
        
        # Stack RGB (Red, Green, Blue)
        # Note: We must stack as BGR because OpenCV saves in Blue-Green-Red order
        # img_stack is R, G, B, N. So:
        # Blue = img_stack[:,:,2] (Wait, check your read logic below)
        
        # Let's be explicit to avoid confusion:
        # earlier in the script: bands = (b, g, r, n)
        # So b=0, g=1, r=2, n=3? NO.
        # Check your simulate_camera_capture return: return (b, g, r, n)
        blue_band = bands[0]
        green_band = bands[1]
        red_band   = bands[2]
        
        # Stack for OpenCV (Blue, Green, Red)
        preview = np.dstack((blue_band, green_band, red_band))
        
        # NORMALIZE & BRIGHTEN
        # Convert to float 0.0 - 1.0
        preview = preview.astype(float) / 65535.0
        
        # CONTRAST STRETCH (The Magic Step)
        # Multiply brightness by 3.5x to fix the "Dark/Trash" look
        preview = preview * 3.5 
        # Clip values that go above 1.0 (so they don't loop back to black)
        preview = np.clip(preview, 0, 1)
        
        # Convert to 8-bit (0-255) for JPG
        preview = (preview * 255).astype(np.uint8)
        
        cv2.imwrite(jpg_path, preview)
        
        print(f"      📦 Packet Saved: .npy (Science) + .jpg (Preview)")

    return img_stack, prediction, status

# --- MAIN LOOP ---
# Run 5 simulations
for i in range(5):
    bands, img_id = simulate_camera_capture()
    
    if bands:
        img, mask, status = process_and_decide(bands, img_id)
        
        # Optional: Visualize the last one just to verify
        if i == 4:
            plt.figure(figsize=(10, 4))
            plt.subplot(1, 2, 1)
            # False Color (NIR, R, G)
            nrg = np.dstack((bands[3], bands[2], bands[1])).astype(float) / 65535.0
            plt.imshow(np.clip(nrg*3, 0, 1))
            plt.title(f"Sensor Input\n(Status: {status})")
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(mask > 0.5, cmap='gray')
            plt.title(f"AI Cloud Mask")
            plt.axis('off')
            plt.show()
    
    time.sleep(1) # Pause between captures