import paho.mqtt.client as mqtt
import subprocess
import time
import os
import sys 
import glob
import re
# --- NEW IMPORTS FOR SINGLE-SHOT CAPTURE ---
import cv2 
# We don't explicitly need numpy here, but cv2 often relies on it.

# --- 1. CONFIGURATION ---
MQTT_BROKER = "broker.hivemq.com"
MQTT_PORT = 1883
AI_TRIGGER_TOPIC = "/smart_sentinel/command/ai_trigger"

# *** IMPORTANT ACI TASK: CHANGE THIS INDEX IF CAMERA FAILS TO OPEN ***
# Confirmed Index is 1, but be aware it might shift to 0 or 2 if other devices connect/disconnect.
CAMERA_INDEX = "1" 

YOLO_ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) # Assumes listener is in /yolov5
TEMP_IMAGE_PATH = os.path.join(YOLO_ROOT_DIR, 'temp_capture.jpg') # Path to save the single captured frame

PERSON_CLASS_ID = '0' # Class ID for 'person' in the standard COCO dataset

is_detecting = False 

# --- HELPER FUNCTIONS ---

def capture_single_frame(index, path):
    """Captures a single frame from the camera and saves it to a file using OpenCV."""
    print(f"INFO: Attempting to capture single frame from camera index {index}...")
    try:
        # Convert index string to integer for cv2.VideoCapture
        cap = cv2.VideoCapture(int(index))
        
        if not cap.isOpened():
            print(f"ERROR: Cannot open camera index {index}. Check if it is in use by other applications.")
            return False
            
        # CRITICAL: Wait for camera to warm up and auto-adjust exposure
        time.sleep(1.5) 
        
        ret, frame = cap.read()
        cap.release() # Release camera immediately after capture
        
        if ret:
            # Save the captured frame to the temporary path
            cv2.imwrite(path, frame)
            print(f"INFO: Single frame successfully saved to {path}")
            return True
        else:
            print("ERROR: Failed to read frame from camera.")
            return False
    except Exception as e:
        print(f"FATAL CAPTURE ERROR: {e}")
        return False


def get_latest_exp_folder():
    """Finds the path to the most recent 'runs/detect/expX' directory."""
    try:
        search_path = os.path.join(YOLO_ROOT_DIR, 'runs', 'detect', 'exp*')
        # Use glob to find all exp folders and max() to get the latest one
        list_of_files = glob.glob(search_path)
        if not list_of_files:
            return None
        # Check if the folder creation time is very recent (e.g., within the last 30 seconds)
        # Note: We keep this logic even though we run fast, just in case of stale folders.
        recent_files = [f for f in list_of_files if os.path.getctime(f) > time.time() - 30]
        if not recent_files:
             # Fallback: just return the latest by time, regardless of how old
             return max(list_of_files, key=os.path.getctime)

        return max(recent_files, key=os.path.getctime)
        
    except Exception as e:
        print(f"ERROR: Could not find latest YOLO output folder: {e}")
        return None

def get_saved_evidence_path(exp_folder):
    """
    Finds the path to the SINGLE most recently created crop.
    """
    # Look in the specific folder for cropped images of the 'person' class ID
    crops_dir = os.path.join(exp_folder, 'crops', 'person')
    
    if os.path.exists(crops_dir):
        # Find the latest JPG file in the crops directory
        list_of_crops = glob.glob(os.path.join(crops_dir, '*.jpg'))
        
        if list_of_crops:
            # Return the latest saved crop (the most recent evidence). 
            # Since we only process one image, this list should only contain one file.
            return max(list_of_crops, key=os.path.getctime)

    print("WARNING: Cropped evidence image not found in the output folder.")
    return None

def check_for_person(exp_folder):
    """
    Analyzes the classification text files in the given folder for the 'person' class ID (0).
    Returns True if a person is detected, False otherwise.
    """
    # This logic now works correctly because --save-txt is re-added to YOLO_COMMAND
    labels_dir = os.path.join(exp_folder, 'labels')
    if not os.path.exists(labels_dir):
        # If labels folder is missing, we cannot classify based on file content.
        print("WARNING: Label folder not found (did YOLO run and save successfully?).")
        return False

    for label_file in os.listdir(labels_dir):
        if label_file.endswith('.txt'):
            try:
                # Check for label files that are not empty (size > 0 bytes)
                file_path = os.path.join(labels_dir, label_file)
                if os.path.getsize(file_path) == 0:
                     print(f"DEBUG: Skipping empty label file: {label_file}")
                     continue

                with open(file_path, 'r') as f:
                    content = f.read()
                    
                    # Check if the content starts with the 'person' ID (0) 
                    # followed by a space (to ensure it's not part of another number)
                    if re.search(r'\b' + re.escape(PERSON_CLASS_ID) + r'\b', content):
                        return True
            except Exception as e:
                print(f"ERROR: Failed to read label file {label_file}: {e}")
                
    return False

# --- MQTT CALLBACKS ---

def on_connect(client, userdata, flags, rc):
    """Callback function when the client connects to the MQTT broker."""
    if rc == 0:
        print(f"DEBUG: Successfully connected to MQTT Broker.")
        client.subscribe(AI_TRIGGER_TOPIC)
        print(f"DEBUG: Subscribed to topic: {AI_TRIGGER_TOPIC}")
    else:
        print(f"FATAL: Connection failed with return code {rc}")
        sys.exit(1)

def on_message(client, userdata, msg):
    """Callback function when a message is received on a subscribed topic."""
    global is_detecting
    
    payload_str = msg.payload.decode()
    print(f"INFO: Received message on {msg.topic}: {payload_str}")
    
    if msg.topic == AI_TRIGGER_TOPIC and payload_str == "ACTIVATE":
        if not is_detecting:
            # We start by capturing a single frame first
            print("--- !!! AI TRIGGER RECEIVED !!! --- Starting single-frame capture...") 
            is_detecting = True
            start_yolo_detection()
        else:
            print("INFO: YOLO is already running. Ignoring redundant trigger.")

# --- CORE LOGIC ---

def start_yolo_detection():
    """Function to execute YOLOv5 detection, check results, and trigger alert."""
    global is_detecting
    
    # 1. Capture a single frame first
    if not capture_single_frame(CAMERA_INDEX, TEMP_IMAGE_PATH):
        is_detecting = False
        print("FATAL: Cannot proceed without a valid image frame. Returning to standby.")
        return

    # 2. Configure YOLO command to run on the single image file
    YOLO_COMMAND = [
        "python", "detect.py", 
        "--weights", "yolov5s.pt", 
        "--source", TEMP_IMAGE_PATH, # <<< NEW SOURCE: single image file
        "--conf", "0.40", 
        "--save-crop", 
        "--save-txt", 
        "--nosave", 
        "--hide-conf", 
        "--hide-labels",
    ]

    process = None
    try:
        # We pass the command as a list and disable shell=True
        print(f"INFO: Executing command: {' '.join(YOLO_COMMAND)}")
        
        # Start YOLOv5
        process = subprocess.Popen(YOLO_COMMAND, shell=False) 
        
        # --- NEW LOGIC: Wait for the subprocess to complete naturally ---
        print("INFO: Waiting for YOLO process to complete inference on single image...")
        process.wait() 
        # The line above waits until detect.py finishes its job and exits.
        
        print("INFO: YOLO process completed successfully.")
        
        # CRITICAL FIX: Give OS a moment to flush buffers and complete file writing (0.5s is safe)
        time.sleep(0.5) 


        # 3. Find the output folder
        latest_folder = get_latest_exp_folder()
        
        if not latest_folder:
            print("WARNING: Could not locate YOLO output folder. Assuming no detection.")
            return

        print(f"INFO: Analyzing results in: {latest_folder}")

        # 4. Check for 'person' (Class ID 0)
        if check_for_person(latest_folder):
            # --- CONDITION MET: PERSON DETECTED ---
            evidence_path = get_saved_evidence_path(latest_folder)
            
            print(">>> ✅ HIGH PRIORITY ALERT: PERSON DETECTED!")
            if evidence_path:
                # Evidence is now a single small JPG crop
                print(f"!!! EVIDENCE SAVED AT: {evidence_path}") 
            else:
                print("!!! WARNING: PERSON DETECTED, but no cropped image evidence was saved. Check YOLO configuration.")
            
            # --- ACTION: Send the final confirmed alert (e.g., via Firebase/LINE) ---
            print("!!! ACTION: Sending Critical Intruder Alert to Firebase/LINE Notify.")
            
        else:
            # --- CONDITION NOT MET: ANIMAL or NONE DETECTED ---
            print(">>> ❌ Classification: Detected ANIMAL, VEHICLE, or NOTHING. Returning to standby.")
            
            # Optional: Delete the output folder to save space if it wasn't a person
            # shutil.rmtree(latest_folder) 

    except subprocess.CalledProcessError as e:
        print(f"ERROR: Subprocess error during YOLO execution: {e}")
    except Exception as e:
        print(f"FATAL: An unhandled error occurred: {e}")
    finally:
        # 5. Clean up the temporary image file
        if os.path.exists(TEMP_IMAGE_PATH):
            os.remove(TEMP_IMAGE_PATH)
            print(f"INFO: Cleaned up temporary image: {TEMP_IMAGE_PATH}")
            
        is_detecting = False
        print("INFO: YOLO Host is now back in standby (listening mode).")

# --- 2. MAIN EXECUTION ---
if __name__ == "__main__":
    print("--- STARTING YOLO LISTENER SERVICE (CONDITIONAL MODE) ---") 
    try:
        print(f"INFO: Using MQTT Broker: {MQTT_BROKER}:{MQTT_PORT}")
        
        client = mqtt.Client()
        client.on_connect = on_connect
        client.on_message = on_message
        
        client.connect(MQTT_BROKER, MQTT_PORT, 5) 
        
        print("INFO: Entering continuous loop. Waiting for trigger...")
        client.loop_forever()
        
    except NameError:
        print("\nFATAL ERROR: paho.mqtt.client is not defined. Did you run 'pip install paho-mqtt' in the correct environment?")
        sys.exit(1)
    except TimeoutError:
        print("\nFATAL ERROR: Connection Timeout. Check your internet connection or MQTT_BROKER address.")
        sys.exit(1)
    except Exception as e:
        print(f"\nFATAL UNHANDLED ERROR: {e}")
        sys.exit(1)