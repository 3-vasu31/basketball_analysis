from utils import read_video, save_video
from trackers import PlayerTracker, BallTracker
from drawers import PlayerTracksDrawer, BallTracksDrawer
from team_assigner import TeamAssigner


def main():

    # read the video
    video_frames= read_video(r"input_videos/video_1.mp4")

    # Initialze the player tracker
    player_tracker=PlayerTracker(r"models/player_detector.pt")

    # Initialize the ball tracker
    ball_tracker = BallTracker(r"models/ball_detector_model.pt")

    # Run Trackers
    player_tracker = player_tracker.get_object_tracks(video_frames,
                                                      read_from_stub=True,
                                                      stub_path= "stubs/player_track_stub.pkl")

    ball_tracks = ball_tracker.get_object_tracks(video_frames,
                                                  read_from_stub = True,
                                                  stub_path= "stubs/ball_track_stub.pkl")
    # Remove wrong detections
    ball_tracks = ball_tracker.remove_wrong_detections(ball_tracks)

    #interpolate missing ball positions
    ball_tracks = ball_tracker.interpolate_ball_positions(ball_tracks)

    #Assign Player Teams
    team_assigner = TeamAssigner()
    player_teams = team_assigner.get_player_teams_accross_frames(video_frames,
                                                                 player_tracker,
                                                                 read_from_stub= True,
                                                                 stub_path="stubs/player_assignment_stub.pkl")
    
    print(player_teams)
    # Drqw output
    # Initialize the player drawer
    player_tracks_drawer= PlayerTracksDrawer()
    # Initialize the ball drawer
    ball_tracks_drawer = BallTracksDrawer()

    # Draw the player tracks on the video frames
    output_video_frames = player_tracks_drawer.draw(video_frames, player_tracker)
    # Draw the ball tracks on the video frames
    output_video_frames = ball_tracks_drawer.draw(output_video_frames, ball_tracks)

    #save the video
    save_video(output_video_frames, "output_videos/output_video.avi")

if __name__ == "__main__":
    main()