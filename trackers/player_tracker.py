from ultralytics import YOLO
import supervision as sv #python library for computer vision to track objects
import sys
sys.path.append("../") #add the parent directory to the path

from utils import read_stub, save_stub

'''A class that handles the player detection and tracking using YOLO and ByteTrack.
This class uses the YOLO object detection with the ByteTrack tracking to maintain consistent
player identities across frames while processing detection in batches.'''
class PlayerTracker:
    def __init__(self,model_path):
        self.model=YOLO(model_path)
        self.tracker=sv.ByteTrack() #multi object tracking algorithm

    def detect_frames(self,frames):
        batch_size=20
        detections=[]
        for i in range(0,len(frames),batch_size):
            batch_frames=frames[i:i+batch_size]
            batch_detections=self.model.predict(batch_frames,conf=0.5) # conf=0.5, set the thresold for detection
            detections+=batch_detections
        return detections
    """
        Get player tracking results for a sequence of frames with optional caching.

        Args:
            frames (list): List of video frames to process.
            read_from_stub (bool): Whether to attempt reading cached results.
            stub_path (str): Path to the cache file.

        Returns:
            list: List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.
        """
    

    def get_object_tracks(self,frames , read_from_stub = False, stub_path = None):
        
        tracks=read_stub(read_from_stub,stub_path)
        if tracks is not None:
            if len(tracks) == len(frames):
                return tracks

        detections=self.detect_frames(frames)
        tracks=[]

        for frame_num,detection in enumerate(detections):
            cls_names=detection.names
            cls_names_inv= {v:k for k,v in cls_names.items()}

            detections_supervision=sv.Detections.from_ultralytics(detection) # from ultralytics detection to supervision detection format

            detection_with_tracks=self.tracker.update_with_detections(detections_supervision)

            # for each frame we now be having a dictionary, key is the id of the player and value is the list of detections
            tracks.append({})
            for frame_detection in detection_with_tracks:
                bbox=frame_detection[0].tolist()
                cls_id=frame_detection[3]
                track_id=frame_detection[4]

                if cls_id == cls_names_inv["Player"]:
                    tracks[frame_num][track_id] = {"box": bbox} # track_id is the key and the value is the bbox

        save_stub(stub_path,tracks)
        return tracks