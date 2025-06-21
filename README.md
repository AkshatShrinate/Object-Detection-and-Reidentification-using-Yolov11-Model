#INTRODUCTION
This project focuses on identifying and tracking players in a football match using a combination of deep learning-based object detection and player re-identification techniques. Leveraging YOLOv11 for real-time object detection, the system accurately identifies players in each frame of a match video, while a custom team-color classifier helps associate each player with their team (red, white, or yellow jerseys). This allows the system to display individual IDs, team affiliations, and track player movements consistently across frames.

To maintain robust tracking, the project integrates the BoT-SORT (Better Object Tracking with Strong ReID) framework, enabling smooth re-identification even in the presence of occlusions or fast movements. Additionally, the project includes a real-time tactical map that visualizes player positions on the field using live positional data. The system outputs an annotated video along with this tactical view, making it a powerful tool for sports analytics and visual review.

#Setup Instructions 
Step 1: Clone the repository
git clone https://github.com/AkshatShrinate/Object-Detection-and-Reidentification-using-Yolov11-Model.git
cd Object-Detection-and-Reidentification-using-Yolov11-Model

Step 2: Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate (for Windows)
source venv/bin/activate (for Mac/Linux)

Step 3: Install the required dependencies
pip install -r requirements.txt

Step 4: Clone BoT-SORT into the project directory
git clone https://github.com/mikel-brostrom/Yolov5_StrongSORT_OSNet.git botsort

Step 5: Ensure required files are present

models/best.pt

assets/15sec_input_720p.mp4

botsort/config/botsort.yaml

Step 6: Run the main script
python main.py
