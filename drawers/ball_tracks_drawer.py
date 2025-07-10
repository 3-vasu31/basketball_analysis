from .utils import draw_triangle
class BallTracksDrawer:
    def __init__(self):
        self.ball_pointer_color = (0, 255, 0)

    def draw(self, video_frames, tracks):
        """
        Draws the ball tracks on the video frames.

        Args:
            video_frames (list): List of video frames to draw on.
            tracks (list): List of dictionaries containing ball tracking information for each frame,
                where each dictionary contains the bounding box of the ball.

        Returns:
            list: List of video frames with drawn ball tracks.
        """
        output_video_frames = []

        for frame_num, frame in enumerate(video_frames):
            output_frame = frame.copy()
            ball_dict = tracks[frame_num]

            for _, track in ball_dict.items():
                bbox = track['bbox']
                if bbox is None:
                    continue
                output_frame = draw_triangle(frame, bbox, self.ball_pointer_color)

            output_video_frames.append(output_frame)

            
        return output_video_frames