# from PIL import Image
# from transformers import CLIPProcessor, CLIPModel
# import cv2
# import sys
# sys.path.append("../")
# from utils import read_stub, save_stub



# class TeamAssigner:
#     '''Assigns players to teams based on their jersey colors using CLIP model.
#         using zero-shot classification.
        
#     '''
#     def __init__( self, 
#                  team_1_class_name = 'white shirt',
#                  team_2_class_name = 'dark blue shirt'):
#         self.team_1_class_name = team_1_class_name
#         self.team_2_class_name = team_2_class_name

#         self.player_team_dict = {}

#     def load_model(self,):
#         self.model = CLIPModel.from_pretrained("patrick-johncyh/fashion-clip")
#         self.processor = CLIPProcessor.from_pretrained("patrick-johncyh/fashion-clip")

#     def get_player_color(self, frame, bbox):
#         '''Extracts the color of the player from the frame using the bounding box.
        
#         Args:
#             frame (PIL.Image): The video frame.
#             bbox (list): The bounding box coordinates [x1, y1, x2, y2].
        
#         Returns:
#             str: The color of the player.
#         '''
#         image = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])] # Extract the region of interest
#         rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

#         pil_image = Image.fromarray(rgb_image)

#         classes = [self.team_1_class_name, self.team_2_class_name]

#         inputs = self.processor(text = classes, images = pil_image, return_tensors="pt", padding=True)

#         outputs = self.model(**inputs)
#         logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
#         probs = logits_per_image.softmax(dim=1)


#         return classes[probs.argmax(dim=1)[0]]

#     def get_player_team(self, frame, player_bbox, player_id):
#         '''Assigns a team to the player based on their jersey color.
        
#         Args:
#             frame (PIL.Image): The video frame.
#             player_bbox (list): The bounding box coordinates [x1, y1, x2, y2].
#             player_id (int): The ID of the player.
        
#         Returns:
#             str: The team assigned to the player.
#         '''
#         if player_id in self.player_team_dict:
#             return self.player_team_dict[player_id]
        
#         player_color = self.get_player_color(frame, player_bbox)
        
#         team_id = 2

#         if player_color == self.team_1_class_name:
#             team_id = 1

#         self.player_team_dict[player_id] = team_id
#         return team_id
    
#     def get_player_teams_accross_frames(self, video_frames, player_tracks, read_from_stub = False, stub_path = None):
#         '''Assigns teams to players across multiple frames.
        
#         Args:
#             video_frames (list): List of video frames.
#             player_tracks (dict): Dictionary containing player tracks.
#             read_from_stub (bool): Whether to read from a stub file.
#             stub_path (str): Path to the stub file if read_from_stub is True.
        
#         Returns:
#             dict: Dictionary containing player IDs and their assigned teams.
#         '''
#         player_assignment = read_stub(read_from_stub, stub_path)
#         if player_assignment is not None:
#             if len(player_assignment) == len(video_frames):
#                 return player_assignment

#         self.load_model()

#         player_assignment = []

#         for frame_num, player_track in enumerate(player_tracks):

#             player_assignment.append({})

#             # Making player_team_dict empty at the interval of 50 frames so that there is reduction for missclassification.
#             if frame_num %50 == 0:
#                 self.player_team_dict = {}
#             for player_id, track in player_track.items():
#                 team = self.get_player_team(video_frames[frame_num], track["bbox"], player_id)
#                 player_assignment[frame_num][player_id] = team
        

#         save_stub(stub_path, player_assignment)
#         return player_assignment

from PIL import Image
from transformers import pipeline
import cv2
import sys
sys.path.append("../")
from utils import read_stub, save_stub

class TeamAssigner:
    '''Assigns players to teams based on their jersey colors using CLIP model.
        using zero-shot classification.

    '''
    def __init__( self,
                  team_1_class_name = 'white shirt',
                  team_2_class_name = 'dark blue shirt'):
        self.team_1_class_name = team_1_class_name
        self.team_2_class_name = team_2_class_name

        self.player_team_dict = {}
        self.classifier = None # Initialize classifier as None

    def load_model(self,):
        # Load the zero-shot image classification pipeline
        self.classifier = pipeline("zero-shot-image-classification", model="patrickjohncyh/fashion-clip")

    def get_player_color(self, frame, bbox):
        '''Extracts the color of the player from the frame using the bounding box.

        Args:
            frame (numpy.ndarray): The video frame (OpenCV format).
            bbox (list): The bounding box coordinates [x1, y1, x2, y2].

        Returns:
            str: The predicted class name (team color description).
        '''
        if self.classifier is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        # Extract the region of interest
        image_roi = frame[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]
        rgb_image_roi = cv2.cvtColor(image_roi, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image_roi)

        candidate_labels = [self.team_1_class_name, self.team_2_class_name]

        # Use the pipeline for classification
        results = self.classifier(pil_image, candidate_labels=candidate_labels)

        # The results are sorted by score, so the first element is the highest probability
        predicted_label = results[0]['label']
        return predicted_label

    def get_player_team(self, frame, player_bbox, player_id):
        '''Assigns a team to the player based on their jersey color.

        Args:
            frame (numpy.ndarray): The video frame (OpenCV format).
            player_bbox (list): The bounding box coordinates [x1, y1, x2, y2].
            player_id (int): The ID of the player.

        Returns:
            int: The team ID assigned to the player (1 for team_1, 2 for team_2).
        '''
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]

        player_color = self.get_player_color(frame, player_bbox)

        team_id = 2 # Default to team 2

        if player_color == self.team_1_class_name:
            team_id = 1

        self.player_team_dict[player_id] = team_id
        return team_id

    def get_player_teams_accross_frames(self, video_frames, player_tracks, read_from_stub = False, stub_path = None):
        '''Assigns teams to players across multiple frames.

        Args:
            video_frames (list): List of video frames (each frame is a numpy.ndarray).
            player_tracks (dict): Dictionary containing player tracks.
            read_from_stub (bool): Whether to read from a stub file.
            stub_path (str): Path to the stub file if read_from_stub is True.

        Returns:
            list: List of dictionaries, where each dictionary contains player IDs and their assigned teams for a frame.
        '''
        player_assignment = read_stub(read_from_stub, stub_path)
        if player_assignment is not None:
            if len(player_assignment) == len(video_frames):
                return player_assignment

        self.load_model() # Load the model when starting team assignment across frames

        player_assignment = []

        for frame_num, player_track in enumerate(player_tracks):

            player_assignment.append({})

            # Making player_team_dict empty at the interval of 50 frames so that there is reduction for missclassification.
            if frame_num % 50 == 0:
                self.player_team_dict = {}
                
            for player_id, track in player_track.items():
                team = self.get_player_team(video_frames[frame_num], track["box"], player_id)
                player_assignment[frame_num][player_id] = team


        save_stub(stub_path, player_assignment)
        return player_assignment