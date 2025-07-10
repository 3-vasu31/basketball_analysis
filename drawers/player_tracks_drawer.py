from .utils import draw_ellipse


class PlayerTracksDrawer:
    
    def __init__(self):
        pass

    def draw(self, video_frames, tracks):
        """
        Draws the player tracks on the video frames.

        Args:
            video_frames (list): List of video frames to draw on.
            tracks (list): List of dictionaries containing player tracking information for each frame,
                where each dictionary maps player IDs to their bounding box coordinates.

        Returns:
            list: List of video frames with drawn player tracks.
        """
        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()

            player_dict= tracks[frame_num]

            # Draw the player tracks on the frame

            for track_id, player in player_dict.items():
                
                frame = draw_ellipse( frame, player['box'], color = (0,0,255), track_id=track_id)

            output_frames.append(frame)


        return output_frames
    