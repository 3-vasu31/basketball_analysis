# Basketball Analysis System

An advanced computer vision system for analyzing basketball gameplay using YOLOv5 and OpenCV. This project tracks players and the ball in real-time, assigns teams, detects ball possession, and calculates team ball control statistics.

## Demo

![Demo Video](demo\output_video-ezgif.com-speed.gif)



## Features

- **Player Detection & Tracking**: Real-time player detection and tracking using YOLOv5
- **Ball Detection & Tracking**: Accurate ball tracking with interpolation for missed detections
- **Team Assignment**: Automatic team identification based on jersey colors using K-Means clustering
- **Ball Possession Detection**: Identifies which player has possession of the ball in each frame
- **Team Ball Control Statistics**: Calculates and visualizes team ball control percentages
- **Visual Annotations**: 
  - Player tracking with team-colored ellipses
  - Ball tracking with triangular markers
  - Real-time ball control statistics overlay

## Project Structure

```
basketball_analysis/
├── main.py                          # Main execution script
├── trackers/                        # Object tracking modules
│   ├── player_tracker.py           # Player detection and tracking
│   └── ball_tracker.py             # Ball detection and tracking
├── drawers/                         # Visualization modules
│   ├── player_tracks_drawer.py     # Player visualization
│   ├── ball_tracks_drawer.py       # Ball visualization
│   ├── team_ball_control_drawer.py # Statistics overlay
│   └── utils.py                    # Drawing utilities
├── team_assigner/                   # Team assignment module
│   └── team_assigner.py            # K-Means based team clustering
├── ball_acquisition_detector/       # Ball possession detection
│   └── ball_acquisition_detector.py
├── utils/                           # Utility functions
│   ├── video_utils.py              # Video I/O operations
│   ├── bbox_utils.py               # Bounding box operations
│   └── stubs_utils.py              # Caching utilities
├── models/                          # YOLOv5 model weights
│   ├── player_detector.pt
│   └── ball_detector_model.pt
├── input_videos/                    # Input video files
├── output_videos/                   # Processed output videos
└── stubs/                           # Cached tracking data
```

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (recommended for faster processing)

### Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/3-vasu31/basketball_analysis.git
   cd basketball_analysis
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv basketball
   basketball\Scripts\activate  # On Windows
   # source basketball/bin/activate  # On Linux/Mac
   ```

3. **Install dependencies**
   ```bash
   pip install opencv-python
   pip install numpy
   pip install ultralytics
   pip install scikit-learn
   pip install pandas
   ```

4. **Download YOLOv8 models**
   - Place your trained player detector model in `models/player_detector.pt`
   - Place your trained ball detector model in `models/ball_detector_model.pt`
   - Alternatively, train your own models using the notebook in `training_notebook/`

## Usage

### Basic Usage

1. **Place your input video** in the `input_videos/` folder

2. **Run the analysis**
   ```bash
   python main.py
   ```

3. **Find the output** in `output_videos/output_video.avi`

### Using Cached Results

The system automatically caches tracking results in the `stubs/` folder. To disable caching and force reprocessing:

```python
# In main.py, change read_from_stub to False
player_tracks = player_tracker.get_object_tracks(
    video_frames,
    read_from_stub=False,  # Set to False
    stub_path="stubs/player_track_stub.pkl"
)
```

### Customization

**Adjust team colors:**
```python
# In main.py
player_tracks_drawer = PlayerTracksDrawer(
    team_1_color=[255, 245, 238],  # Light color
    team_2_color=[128, 0, 0]       # Dark red
)
```

**Change ball marker color:**
```python
# In drawers/ball_tracks_drawer.py
self.ball_pointer_color = (0, 255, 0)  # Green (BGR format)
```

## How It Works

1. **Video Processing**: Reads input video frame by frame
2. **Object Detection**: Uses YOLOv5 models to detect players and ball
3. **Tracking**: Maintains object IDs across frames
4. **Team Assignment**: Clusters players into teams based on jersey colors
5. **Ball Possession**: Calculates distance between ball and players to determine possession
6. **Visualization**: Draws tracking information and statistics on frames
7. **Output**: Saves annotated video with all visualizations

## Training Custom Models

The project includes a Jupyter notebook for training custom YOLOv8 models:

```
training_notebook/basketball_object_detection_training.ipynb
```

Follow the notebook to:
- Prepare your dataset
- Train player and ball detection models
- Export trained models to the `models` folder

## Performance Optimization

- **Use stubs**: Enable `read_from_stub=True` to cache results
- **GPU acceleration**: Ensure CUDA is properly installed for faster inference
- **Reduce video resolution**: Process at lower resolution for faster processing
- **Batch processing**: Adjust batch size in tracker initialization

## Troubleshooting

**Import errors:**
- Ensure all modules have `__init__.py` files
- Check that you're running from the project root directory

**Model not found:**
- Verify model paths in `main.py`
- Ensure `.pt` files are in the `models` folder

**Memory issues:**
- Process shorter video clips
- Reduce video resolution
- Enable frame skipping

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Acknowledgments

- YOLOv5 by Ultralytics
- OpenCV community
- Basketball dataset contributors

## Contact

**Author**: 3-vasu31  
**GitHub**: [https://github.com/3-vasu31](https://github.com/3-vasu31)  
**Project Link**: [https://github.com/3-vasu31/basketball_analysis](https://github.com/3-vasu31/basketball_analysis)

---

