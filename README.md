
🛡️ AI Face Security System (Python)
An intelligent surveillance system that uses Deep Learning to recognize faces through your webcam.

⚙️ How it works:
Scenario	Logic	Siren Status
Known Person---->Face matches your saved photos	❌ OFF
Unknown Person ---->Face detected but no match found	✅ ON (ALARM)
No Person ----->	No face detected in frame	❌ OFF
1. Prerequisites
Operating System: Windows 10/11
Hardware: Integrated Webcam or USB Camera
Python Version: Python 3.10.x or 3.11.x (Recommended)
Siren File: An audio file named siren.mp3 in the main folder.
2. Installation
Open your terminal (Command Prompt or VS Code Terminal) and run these commands in order:

Step A: Clean old versions
Bash

python -m pip uninstall opencv-python -y
Step B: Install required libraries
Bash

python -m pip install opencv-contrib-python pygame numpy
3. Project Structure
Ensure your folder looks exactly like this:

text

face_detection/
├── known_faces/        # Folder: Stores your captured photos
├── models/             # Folder: Stores AI models (Auto-downloaded)
├── capture_faces.py    # Script: Run this first to save your face
├── main.py             # Script: The main security system
└── siren.mp3           # Audio: Your alarm sound
4. How to Use
Phase 1: Capture Your Face
Run the capture script:
Bash

python capture_faces.py
A window will open. Look at the camera.
When you see a Green Box, press SPACEBAR.
Capture 5 photos.
Tip: Change your expression slightly for each (smile, serious, neutral) to improve accuracy.
The program will close automatically once finished.
Phase 2: Start Security
Run the main system:
Bash

python main.py
The system will load your saved photos and start the camera.
Green Box (KNOWN): Safe.
Red Box (UNKNOWN): The siren will trigger after 3 seconds of continuous detection.
5. Live Controls (While running main.py)
If the system is too strict (calling you unknown) or too lenient (calling strangers known), use these keys:

+ (Plus): Make the system Stricter (Harder to match).
- (Minus): Make the system Lenient (Easier to match).
q (Quit): Stop the system and close the camera.
6. Troubleshooting
AttributeError: module 'cv2' has no attribute 'face': This means you have the wrong OpenCV version. Run the "Installation" commands above again.
Siren not playing: Ensure your file is named exactly siren.mp3 and your speakers are turned on.
Low Accuracy: Ensure you are in a well-lit room. Shadows on the face can confuse the AI.
7. Technical Versions Used
Python: 3.11.9
OpenCV-Contrib-Python: Latest (4.x)
Detection Model: Caffe SSD (ResNet-10)
Recognition Model: SFace (Deep Learning Embeddings)
Audio Engine: Pygame 2.x
  An  Priorix  Product
