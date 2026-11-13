from flask import Flask, render_template, request, jsonify
import cv2
import numpy as np
import base64
from utils.detector import DrowsinessDetector
from PIL import Image
import io
import time

app = Flask(__name__)
detector = DrowsinessDetector()

start_time = time.time()
session_stats = {
    'total_alerts': 0,
    'total_yawns': 0,
}

# State tracking to prevent counting the same alert multiple times
last_status = "ALERT"

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    global last_status
    data = request.json
    if 'image' not in data:
        return jsonify({'error': 'No image provided'}), 400
        
    try:
        # Decode base64 image
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        # Process frame
        result = detector.process_frame(frame)
        
        # Status change tracking for stats
        current_status = result['status']
        if current_status != last_status:
            if current_status == 'DROWSY':
                session_stats['total_alerts'] += 1
            elif current_status == 'YAWN':
                session_stats['total_yawns'] += 1
            last_status = current_status
            
        return jsonify(result)
    except Exception as e:
        print(f"Error processing frame: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/set_model', methods=['POST'])
def set_model():
    data = request.json
    model_type = data.get('model_type', 'custom_cnn')
    try:
        detector.load_models(model_type)
        return jsonify({"success": True, "message": f"Switched to {model_type}"})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@app.route('/stats', methods=['GET'])
def stats():
    duration_seconds = int(time.time() - start_time)
    minutes, seconds = divmod(duration_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return jsonify({
        'total_alerts': session_stats['total_alerts'],
        'total_yawns': session_stats['total_yawns'],
        'session_duration': f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
