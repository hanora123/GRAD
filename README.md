GRAD - Ground-based Road Analysis and Detection
Overview
GRAD is an advanced traffic analysis system that combines computer vision, object detection, tracking, and trajectory prediction to analyze traffic patterns from video footage. The system processes video input from traffic cameras and provides detailed analysis including object detection, tracking, speed estimation, trajectory prediction, and visualization in both camera view and bird's eye view (BEV).

Key Features
Object Detection: Identifies and classifies vehicles, pedestrians, and other road users
Multi-Object Tracking: Tracks objects across video frames with consistent IDs
Trajectory Prediction: Forecasts future paths of moving objects
Speed Estimation: Calculates object speeds in real-world units
Bird's Eye View Transformation: Converts camera perspective to top-down view
Traffic Pattern Analysis: Identifies common movement patterns and anomalies
Visualization Tools: Displays results in both camera view and bird's eye view
Project Structure
plaintext


Requirements
OS: Windows / Linux / Mac
Python: 3.8.1 or above
OpenCV: 4.7.0 or above
NumPy: 1.23.5 or above
PyTorch (for YOLOv5)
Installation
Clone the repository

bash
Run
git clone https://github.com/yourusername/GRAD.gitcd GRAD
Install YOLOv5 requirements

bash
Run
cd code/YOLOv5pip install -r requirements.txtpip install -r requirements_trajectory.txt
Usage
Bird's Eye View Calibration
Before running the main system, you need to calibrate the bird's eye view transformation. Two methods are provided:

Manual Calibration
bash
Run
# Navigate to the Birds-Eye-View-Calibration directorycd Birds-Eye-View-Calibrationpython Calib_GrndPlane.py
This process involves:

Background extraction
ROI determination
Ground plane selection
Aspect ratio refinement
Satellite-based Calibration
bash
Run
# Navigate to the Birds-Eye-View-Calibration directorycd Birds-Eye-View-Calibrationpython Calib_SatFeature.py
This process requires a satellite image of the location and involves:

Background extraction
ROI determination
Point identification in satellite image and video
Aspect ratio refinement
Refer to Guide.pdf in the Birds-Eye-View-Calibration directory for detailed instructions.

Running Object Detection with Trajectory Prediction
bash
Run
# Navigate to the YOLOv5 directorycd code/YOLOv5# Run detection with trajectory predictionpython detect_trajectory.py --source path/to/video.mp4 --weights yolov5l.pt --history-size 30 --future-steps 10
Command Line Arguments
--source: Path to input video file or camera index
--weights: Path to YOLOv5 model weights
--history-size: Number of frames to keep in trajectory history (default: 30)
--future-steps: Number of steps to predict into future (default: 10)
--conf-thres: Confidence threshold for detections (default: 0.25)
--iou-thres: IoU threshold for NMS (default: 0.45)
--view-img: Display results in real-time
--custom-output: Custom output path for results
Technical Details
Trajectory Prediction
The trajectory prediction system uses a combination of techniques:

Kalman Filtering: For smooth tracking and velocity estimation
Movement Pattern Analysis: Analyzes common movement patterns for different object classes
Location History: Maintains a heat map of common paths in the scene
Weighted Trajectory Prediction: Combines straight-line and turning predictions based on historical patterns
Configuration
The system uses JSON configuration files to store settings for:

Object real-world sizes
Visualization settings
Speed calculation parameters
Tracking parameters
Video output settings
Example configuration is available in data/Leeds/config.json.

Output
The system generates several outputs:

Processed video with detection boxes and trajectory predictions
Bird's eye view visualization
Speed and movement statistics
Trajectory data
Troubleshooting
If you encounter issues with trajectory prediction, ensure that:

The SORT tracker is properly initialized
The TrackPredictor class has all required methods implemented
The calibration files are correctly set up for your video source
The configuration parameters match your scene characteristics
For best results, use videos with stable camera positions and good lighting conditions.

References
Rezaei, M., Azarmi, M., & Mir, F.M.P. (2023). "3D-Net: Monocular 3D object recognition for traffic monitoring." Expert Systems with Applications, 227, p.120253.
Rezaei, M., & Azarmi, M. (2020). "Deepsocial: Social distancing monitoring and infection risk assessment in COVID-19 pandemic." Applied Sciences, 10(21), p.7514.
License
This project is licensed under the terms included in the LICENSE file.

Acknowledgements
YOLOv5 by Ultralytics
SORT tracking algorithm
OpenCV library
Contact
For questions or support, please open an issue in the GitHub repository.