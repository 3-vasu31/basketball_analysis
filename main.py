from utils import read_video, save_video
from trackers import PlayerTracker
def main():

    # read the video
    video_frames= read_video(r"input_videos\video_1.mp4")

    # Initialze the player tracker
    player_tracker=PlayerTracker(r"models/player_detector.pt")

    # Run Trackers
    player_tracker = player_tracker.get_object_tracks(video_frames,
                                                      read_from_stub=True,
                                                      stub_path= "stubs/palyer_track_stub.pkl")

    print(player_tracker)

    #save the video
    save_video(video_frames,"output_videos\output_video.avi")

if __name__ == "__main__":
    main()