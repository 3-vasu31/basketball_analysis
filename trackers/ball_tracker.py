from ultralytics import YOLO
import sys
import supervision as sv
import numpy as np
import pandas as pd
sys.path.append("..")  
from utils import read_stub, save_stub
class BallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames):
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            batch_frames = frames[i:i+batch_size]
            batch_detections = self.model.predict(batch_frames, conf = 0.5)
            detections += batch_detections
        return detections
    
    def get_object_tracks(self, frames, read_from_stub = False, stub_path = None):

        tracks = read_stub( read_from_stub, stub_path )
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks
            
        detections = self.detect_frames(frames)
        tracks = []

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v:k for k,v in cls_names.items()}

            detection_supervision = sv.Detections.from_ultralytics(detection)
            tracks.append({})
            choosen_bbox = None
            max_confidence = 0
            for frame_detection in detection_supervision:
                bbox = frame_detection[0].tolist()
                cls_id = frame_detection[3]
                confidence = frame_detection[2]

                if cls_id == cls_names_inv['Ball']:
                    if max_confidence < confidence:
                        max_confidence = confidence
                        choosen_bbox = bbox

            if choosen_bbox is not None:
                tracks[frame_num][1] = {
                    'bbox': choosen_bbox
                }
            
        save_stub(stub_path, tracks)
        return tracks

    def remove_wrong_detections(self, ball_positions):
        '''This function removes detections that are too far from the last good detection.
        It is used to filter out false positives in the ball tracking.
        ball_positions: is a list of dictionaries, where each dictionary contains the bounding box of the ball in a frame.
        The structure of ball_positions is as follows:
        ball_positions = [
            {1: {'bbox': [x1, y1, x2, y2]}},  # Frame 0
            {1: {'bbox': [x1, y1, x2, y2]}},  # Frame 1
            ...
        ]
        where 1 is the track ID for the ball.
        maximum_allowed_distance:  is the maximum distance in pixels that a detection can be from the last good detection.
        last_good_from_index: is the index of the last good detection.'''
        maximum_allowed_distance = 25
        last_good_from_index = -1

        for i in range(len(ball_positions)):
            #It safely retrieves the bounding box ('bbox') for the ball (track ID 1) in frame i. If the ball or its bounding box is missing, it returns an empty list instead of raising an error.
            current_bbox = ball_positions[i].get(1, {}).get('bbox', [])

            if len(current_bbox) == 0:
                continue

            if last_good_from_index == -1:
                last_good_from_index = i
                continue

            last_good_box = ball_positions[last_good_from_index].get(1, {}).get('bbox', [])
            frame_gap = i - last_good_from_index
            adjusted_max_distance = maximum_allowed_distance * frame_gap

            #calculate the distance between the last good detection and the current detection
            if np.linalg.norm(np.array(last_good_box[:2]) - np.array(current_bbox[:2])) > adjusted_max_distance:
                ball_positions[i] = {} # if the distance is too large, we remove the detection
            else:
                last_good_from_index = i

        return ball_positions

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions in the tracking data.
        Args:
            ball_positions (list): List of dictionaries containing ball tracking information for each frame.
        Returns:
            list: List of dictionaries with interpolated ball positions.
        """
        ball_positions = [ x.get(1,{}).get('bbox', []) for x in ball_positions]
        df_ballpositions = pd.DataFrame(ball_positions, columns=['x1', 'y1', 'x2', 'y2'])

        #Interpolate missing values
        df_ball_positions = df_ballpositions.interpolate()
        df_ball_positions = df_ball_positions.bfill()

        ball_positions = [ {1: {'bbox': x}} for x in df_ball_positions.to_numpy().tolist()]

        return ball_positions
