from OCC.Display.SimpleGui import init_display
import os

from FeatureRecognition.feature_recognition import FeatureRecognition, FeatureLibrary
from FeatureRecognition.part_visualizer_occ import PartVisualizer_occ
from FeatureRecognition.part_vizualizer_plotly import Part_Visualizer
from FeatureRecognition.aag_builder import AAGBuilder_2D, AAGBuilder_3D
from FeatureRecognition.geometry_analysis import (
    load_step_file,
    analyze_shape,
)

from SetupPlanning.TADs import TAD_Extraction

def main():
    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Load STEP file
    my_shape = load_step_file(os.path.join("STEPFiles", "Part2.stp"))
    if not my_shape:
        return


    # Feature Recognition
    print("\nFEATURE RECOGNITION")
    recognizer = FeatureRecognition(my_shape)
    features = recognizer.identify_features()
    # stats
    print(f"Detected {len(features)} features")
    for f in features:
        print(f["feature_type"], f["node_indices"])


    #TAD
    extractor = TAD_Extraction(my_shape)
    extractor.print_tad_table()
    extractor.print_grouped_tads()


    # Visualize results
    print("\n VISUALIZATION")
    choice = input(
            "Visualize: "
            "(0) Features,"
            "(1) TADs 3D Volumetric, "
            "[0/1]: "
        )


    if choice in ["0"]:  # features
        recognizer.visualize_features_3d(
            show_mesh=True,
            show_face_centers=True,
            show_edges=True
        )



    start_display()

if __name__ == "__main__":
    main()
