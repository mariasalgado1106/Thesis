from OCC.Display.SimpleGui import init_display
import os

from FeatureRecognition.feature_recognition import FeatureRecognition, FeatureLibrary
from FeatureRecognition.part_visualizer_occ import PartVisualizer_occ
from FeatureRecognition.part_vizualizer_plotly import Part_Visualizer
from FeatureRecognition.aag_builder import AAGBuilder_2D, AAGBuilder_3D
from FeatureRecognition.geometry_analysis import (
    load_step_file,
    analyze_shape,
    print_face_analysis_table,
    print_edge_analysis_table
)

def main():
    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Load STEP file
    my_shape = load_step_file(os.path.join("STEPFiles", "Part3.stp"))
    if not my_shape:
        return


    # Feature Recognition
    print("\nPART 3: FEATURE RECOGNITION")
    recognizer = FeatureRecognition(my_shape)
    features = recognizer.identify_features()
    # stats
    print(f"Detected {len(features)} features")
    for f in features:
        print(f["feature_type"], f["node_indices"])