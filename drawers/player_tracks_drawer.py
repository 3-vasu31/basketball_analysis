from .utils import draw_ellipse


class PlayerTracksDrawer:
    
    def __init__(self, team_1_color = [255, 245, 238], team_2_color = [128, 0, 0]):
        '''Initialize the PlayerTracksDrawer with team colors.'''
        
        self.default_player_team_id = 1
        
        self.team_1_color = team_1_color
        self.team_2_color = team_2_color

    def draw(self, video_frames, tracks, player_assignment):
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
            
            player_assignment_for_frame = player_assignment[frame_num]

            # Draw the player tracks on the frame

            for track_id, player in player_dict.items():
                team_id = player_assignment_for_frame.get(track_id, self.default_player_team_id)
                if team_id == 1:
                    color = self.team_1_color
                else:
                    color = self.team_2_color
                    
                frame = draw_ellipse( frame, player['box'], color = color, track_id=track_id)

            output_frames.append(frame)


        return output_frames
    