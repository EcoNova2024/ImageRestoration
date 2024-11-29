import base64
from flask import Flask, request, send_file, jsonify
import cv2
import numpy as np
from io import BytesIO

app = Flask(__name__)

# Function to decode Base64 image to OpenCV image
def decode_base64_image(base64_str):
    img_data = base64.b64decode(base64_str)
    nparr = np.frombuffer(img_data, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

# Function to encode OpenCV image to Base64 (for optional use)
def encode_image_to_base64(img):
    _, buffer = cv2.imencode('.jpg', img)
    img_bytes = buffer.tobytes()
    base64_str = base64.b64encode(img_bytes).decode('utf-8')
    return base64_str


# Enhanced damage detection using edge detection (Canny)
def detect_damage_by_edges(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=100, threshold2=200)
    return edges

# Inpainting to fill damaged areas
def restore_image(image, mask):
    inpainted_image = cv2.inpaint(image, mask, 3, flags=cv2.INPAINT_TELEA)
    return inpainted_image

# Apply Non-Local Means Denoising to smooth out areas and restore textures
def denoise_and_refine(image):
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image

# Function to process and restore the image
def restore_table_image(image):
    # Detect damage using edge detection
    mask = detect_damage_by_edges(image)
    
    # Inpaint the damaged areas using Telea inpainting
    inpainted_image = restore_image(image, mask)
    
    # Denoise and refine the inpainted image
    refined_image = denoise_and_refine(inpainted_image)
    
    return refined_image

# Route to handle the image upload and restoration
@app.route('/restore-image', methods=['POST'])
def restore_image_api():
    data = request.get_json()
    
    if 'image' not in data:
        return jsonify({"error": "No image provided"}), 400
    
    base64_image = data['image']
    
    try:
        # Decode the Base64 string to an OpenCV image
        image = decode_base64_image(base64_image)
        
        if image is None:
            return jsonify({"error": "Invalid image format or corrupted image"}), 400
        
        # Process and restore the image
        restored_image = restore_table_image(image)
        
        # Encode the restored image to JPG format
        _, img_bytes = cv2.imencode('.jpg', restored_image)
        
        # Convert image bytes to a BytesIO object
        img_io = BytesIO(img_bytes.tobytes())
        
        # Return the restored image as a response
        return send_file(img_io, mimetype='image/jpeg', as_attachment=True, download_name="restored_image.jpg")
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
