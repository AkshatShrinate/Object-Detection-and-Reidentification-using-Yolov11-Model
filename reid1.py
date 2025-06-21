import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches


def detect_team(crop):
    #Detect team color from cropped images of the player using the jersey. This allows to code the player ID with colour of the jersey 
    if crop is None or crop.size == 0:
        return "?"

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    #Masking of the cropped images from each frame of the video is done to identify the red,white and yellow regions/jersey. 
    red1 = cv2.inRange(hsv, (0, 50, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 50, 50), (180, 255, 255))
    red_mask = red1 | red2

    white_mask = cv2.inRange(hsv, (0, 0, 200), (180, 40, 255))
    yellow_mask = cv2.inRange(hsv, (20, 100, 100), (35, 255, 255))
    #For each cropped image, we find the percent of pixels for each colour this helps to identify the team color 
    red_pct = np.sum(red_mask > 0) / crop.size
    white_pct = np.sum(white_mask > 0) / crop.size
    yellow_pct = np.sum(yellow_mask > 0) / crop.size

    if red_pct > 0.01:
        return "R"
    elif white_pct > 0.01:
        return "W"
    elif yellow_pct > 0.01:
        return "Y"
    return "?"


def team_color(team):
    #The team code such as R,W or Y is then coded to the color
    return {
        "R": "#ff4c4c",    # Red for red team
        "W": "#cccccc",    # White for white team
        "Y": "#f8e71c",    # Yellow for referee
        "?": "#888888"     # Unknown
    }.get(team, "#888888") #the code kept crashing during the coding without the .get(), which is just a default case to handle exception cases where ? fails as a default 


def setup_tactical_map(frame_shape):
    #To identify the positions of each team player, I created a live matplotlib tactical view.
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 6))
    #Setting width and height 
    ax.set_xlim(0, frame_shape[1])
    ax.set_ylim(frame_shape[0], 0)
    ax.axis('off')
    return fig, ax


def update_tactical_map(ax, positions):
    #After each frame, there is a need to update player positions on the tactical map
    ax.clear()
    ax.set_xlim(0, 1280)
    ax.set_ylim(720, 0)
    ax.axis('off')
    ax.set_title("Tactical Map - Live")
    #I Chose to move forward with a square dots based tactical map of the field 
    for cx, cy, label, color in positions:
        dot = patches.RegularPolygon((cx, cy), numVertices=4, radius=7, orientation=np.pi / 4, color=color)
        ax.add_patch(dot)
        ax.text(cx, cy - 12, label, ha='center', va='bottom', fontsize=8, color='black', weight='bold')

    plt.pause(0.001)


class GlobalTracker:
    #This is a multi-object tracker used to keep track of the assignment and maintaining of the unique IDs given to players on the video feed
    def __init__(self, dist_thresh=60): 
        self.players = {}
        self.next_id = 0
        self.dist_thresh = dist_thresh #this is a distance threhold set to make sure that the players keep same id through different frames of the video

    def _euclidean(self, a, b): #this is calculating euclidean distance of a player between positions a and b of two frames of a video 
        return np.linalg.norm(np.array(a) - np.array(b))

    def match_or_create(self, det_center, team):
        #The function Takes input of a player, its current location and the team it belongs to. 
        # It determines whether it is the same player seen in a previous frame by using the euclidean distance or a new ID is to be issued
        best_gid = None                  #best match player id found, set to None initially 
        best_dist = float("inf")         #smallest distance of the other already identified players from the cropped player position 

        for gid, info in self.players.items():
            if info["team"] != team:     #only checking players with the same team     
                continue
            dist = self._euclidean(det_center, info["last_pos"])
            if dist < self.dist_thresh and dist < best_dist:
                best_dist = dist
                best_gid = gid

        if best_gid is not None:
            self.players[best_gid]["last_pos"] = det_center
        else:
            best_gid = self.next_id
            self.players[best_gid] = {"team": team, "last_pos": det_center}
            self.next_id += 1

        return best_gid #the returned ID for the cropped player from the video frame 