from typing import List, Dict, Any
from aag_builder import AAGBuilder_2D, AAGBuilder_3D

class FeatureRecognition_OpenCV:
    def __init__(self):
        self.aag = AAGBuilder_3D()



class FeatureRecognizer:
    def __init__(self, subgraphs_info):
        self.subgraphs_info = subgraphs_info
        self.recognized_features = []
        self.processed_subgraphs = []

    def recognize_all_features (self):
        self.recognize_holes()
        # self.recognize_pockets()

        return self.recognized_features

    def recognize_holes(self):
        for subgraph in self.subgraphs_info:
            idx = subgraph['subgraph_idx']
            if idx in self.processed_subgraphs:
                continue  # Skip if we already did this

            face_types = subgraph['face_types']
            faces = subgraph['nodes']
            n_faces = subgraph['n_faces']
            n_concave = subgraph['n_concave']
            cyl_count = face_types.count("Cylinder")
            plane_count = face_types.count("Plane")


            #Computer Vision -> Open Cv -> just look at graph and recognize the feature
            # this would be more innovative


            #Through Hole
            #Only 1 node (face) and cylindrical and 0 concave edges
            if n_faces >= 1 and cyl_count == n_faces and n_concave == 0:
                print (f"Through Hole at subgraph {idx}")
                self.recognized_features.append(("feat_hole_through", faces))
                self.processed_subgraphs.append(idx)

            #Blind Hole
            #2 nodes (1 cylindrical and 1 plane); 1 concave edge
            elif (n_faces >= 2 and cyl_count >= 1 and plane_count == 1 and
                  (cyl_count + plane_count == n_faces)):
                print(f"Blind Hole at subgraph {idx}")
                self.recognized_features.append(("feat_hole_blind", faces))
                self.processed_subgraphs.append(idx)

