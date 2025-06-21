#Imported Dependencies
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from yolo_detector import YoloDetector
from reid1 import GlobalTracker, detect_team, team_color, setup_tactical_map, update_tactical_map

# Paths to the model, video to be analysed and the output file to be created
MODEL_PATH = "models/best.pt"
VIDEO_PATH = "assets/15sec_input_720p.mp4"
OUTPUT_PATH = "output.mp4"

def main():
    detector = YoloDetector(MODEL_PATH, confidence=0.3)
   
    tracker = GlobalTracker(dist_thresh=60) #the distance threshold is set at 60px

    cap = cv2.VideoCapture(VIDEO_PATH) 
    if not cap.isOpened():                  #in case the video is not present or accessible 
        print("Failed to open video.")
        return
    #The width, height and fps of the input video is taken as input 
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (width, height))

    ret, frame = cap.read()
    if not ret:                             #ensuring the video is not empty and has frames that can be analysed
        print("No frame to start.")
        return 
                                            #Setting up the live tactical map figure
    fig, ax = setup_tactical_map(frame.shape)

    while True:
        ret, frame = cap.read()            #Until no next frame present 
        if not ret:
            break

        results = detector.track(frame, tracker_config="botsort.yaml")  #result is every person detected in a frame
        positions = []                  #To store details of every result (player) detected over a frame 

        for result in results:
            if not result or not result.boxes:           #to skip a frame if no box detected
                continue

            for box in result.boxes:                    
                cls_id = int(box.cls[0])                #class_id helps to analyse if detected object is a player or not
                label = result.names.get(cls_id, "")    
                if label != "player":                  #To only detect players
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0])      #the coordinates of the player is stored and crop is done 
                crop = frame[y1:y2, x1:x2]
                team = detect_team(crop)                    #to detect the team of the player in the crop
                center = ((x1 + x2) // 2, (y1 + y2) // 2)

                gid = tracker.match_or_create(center, team)

                label_text = f"{gid}:{team}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label_text, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (192, 192, 192), 2)

                positions.append((center[0], center[1], f"{team}{gid}", team_color(team)))

        out.write(frame)
        cv2.imshow("Live Frame", frame)
        update_tactical_map(ax, positions)

        if cv2.waitKey(1) & 0xFF in [27, ord("q")]:           #to exit the GUI windows
            break
    #Shut down the OpenCV windows
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    plt.ioff()
    plt.close()


if __name__ == "__main__":
    main()
