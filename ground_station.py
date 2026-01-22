import os
import numpy as np
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
DOWNLINK_DIR = "to_downlink"

def analyze_packet(filename):
    path = os.path.join(DOWNLINK_DIR, filename)
    
    # 1. Load the Scientific Data (4 Bands: Red, Green, Blue, NIR)
    # Shape is (384, 384, 4)
    data = np.load(path)
    
    # Unpack bands (Remember: In flight software we stacked them as R, G, B, N)
    # Wait! Let's check flight_software.py: "img_stack = np.dstack((r, g, b, n))"
    # So: 0=Red, 1=Green, 2=Blue, 3=NIR
    r = data[:, :, 0]
    g = data[:, :, 1]
    b = data[:, :, 2]
    nir = data[:, :, 3]
    
    # 2. Prepare Visualization
    # Normalize to 0-1 range for plotting
    r_norm = r.astype(float) / 65535.0
    g_norm = g.astype(float) / 65535.0
    b_norm = b.astype(float) / 65535.0
    nir_norm = nir.astype(float) / 65535.0
    
    # --- VIEW 1: TRUE COLOR (What humans see) ---
    # Stack Red, Green, Blue
    true_color = np.dstack((r_norm, g_norm, b_norm))
    true_color = np.clip(true_color * 3.5, 0, 1) # Brighten it up
    
    # --- VIEW 2: THE TRUTH DETECTOR (NIR Band) ---
    # Snow absorbs NIR (Dark), Clouds reflect NIR (Bright)
    nir_view = np.clip(nir_norm * 3.5, 0, 1)

    # --- VIEW 3: NASA FALSE COLOR (Standard Infrared) ---
    # We map: NIR->Red, Red->Green, Green->Blue
    # This makes Vegetation = RED
    # This makes Clouds = WHITE
    # This makes Snow = CYAN (Blue/Green)
    false_color = np.dstack((nir_norm, r_norm, g_norm))
    false_color = np.clip(false_color * 3.5, 0, 1)

    # 3. Plot them side-by-side
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(true_color)
    ax[0].set_title(f"True Color (RGB)\n(Snow & Clouds look White)")
    ax[0].axis('off')
    
    ax[1].imshow(nir_view, cmap='gray')
    ax[1].set_title(f"NIR Band (The Truth)\n(Snow=Dark, Clouds=Bright)")
    ax[1].axis('off')
    
    ax[2].imshow(false_color)
    ax[2].set_title(f"False Color Composite\n(Snow=Cyan, Clouds=White, Trees=Red)")
    ax[2].axis('off')
    
    plt.suptitle(f"Packet Analysis: {filename[:20]}...", fontsize=14)
    plt.tight_layout()
    plt.show()

# --- MAIN LOOP ---
print(f"📡 GROUND STATION: Listening to {DOWNLINK_DIR}...")
files = [f for f in os.listdir(DOWNLINK_DIR) if f.endswith('.npy')]

if not files:
    print("   ❌ No packets found. Run flight_software.py first!")
else:
    print(f"   ✅ Received {len(files)} packets. Decoding...")
    for f in files:
        analyze_packet(f)