# face-recognition-attendance

## Overview

This project requires several Python packages, which are listed in the `requirements.txt` file. Follow the instructions below to set up your development environment.

## Prerequisites

- Python 3.10 or higher
- `pip` (Python package installer)

## Installation Instructions

1. **Clone the repository** (if you haven't already):
   ```bash
   git clone https://github.com/mniranjan1/face-recognition-attendance.git
   cd face-recognition-attendance
    ```
2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate 
    ```
3. **Install the required packages**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install the EVM Package**:
   ```bash
   pip install EVM-0.1.2
   ```
5. **Load the models by copying to models/ folder**

6. **Run the application**:
    - Running the image capture:
    ```bash
    python face_recog_img.py --image_path <path_to_image>
    ```
    - Running the video capture:
    ```bash
    python face_recog_pytorch_final.py
    ```
    -Running whole face recognition with attendance:
    ```bash
    python capture.py
    ```
    Follow the instructions in the window appeared after running it.