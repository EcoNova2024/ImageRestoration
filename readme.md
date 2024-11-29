# Image Restoration API

This is a simple Flask-based API designed to process and restore damaged images. The API accepts an image in Base64 format, processes it by detecting damaged areas, inpaints those areas, and then returns the restored image directly as a response.

## Features:

- **Damage Detection**: Uses edge detection (Canny method) to detect areas of the image that are likely damaged.
- **Inpainting**: Fills the damaged areas using OpenCV’s inpainting methods (Telea).
- **Denoising**: Applies Non-Local Means Denoising to smooth out restored areas and improve image quality.
- **Direct Image Response**: The API responds with the restored image in the form of a direct image (JPEG format).

## Requirements

- Python 3.x
- Flask
- OpenCV
- NumPy


