import os
from OCC.Display.SimpleGui import init_display
from geometry_analysis import (load_step_file, analyze_shape, print_face_analysis_table, print_edge_analysis_table)
from aag_builder import AAGBuilder_2D
from feature_recognition import FeatureRecognizer
from part_visualizer import PartVisualizer


def main():
    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Load STEP file
    my_shape = load_step_file(os.path.join("STEPFiles", "example_thoughhole.stp"))

    if not my_shape:
        return

    # PART 1: Geometry Analysis
    print("\n PART 1: GEOMETRY ANALYSIS")
    all_faces, face_data_list, analyser, all_edges, edge_data_list = analyze_shape(my_shape)
    print_face_analysis_table(all_faces, face_data_list)
    print_edge_analysis_table(all_edges, edge_data_list)


    # PART 2: Build AAG
    print("\n PART 2: AAG CONSTRUCTION")
    builder = AAGBuilder_2D(my_shape)
    subgraphs_info = builder.analyse_subgraphs()

    # PART 3: Feature Recognition
    print("\n PART 3: FEATURE RECOGNITION")
    recognizer = FeatureRecognizer(subgraphs_info)
    features = recognizer.recognize_all_features()
    '''
    # Print statistics --> feature count'''

    # Visualize results
    print("\n VISUALIZATION")
    choice = input("Visualize: (1) Edge Types, (2) Features, (3) Both, (4) AAG + Subgraphs [1/2/3/4]: ")
    viz = PartVisualizer(builder, recognizer)

    if choice in ["1", "3"]:
        viz.display_base_shape(display, transparency=0.6) 
        viz.visualize_edges(display)
    '''
    if choice in ["2", "3"]:
        viz.display_base_shape(display, transparency=0.0) 
        viz.visualize_features(display)
    '''
    if choice == "4":
        builder.visualize_aag()

    '''if choice not in ["1", "2", "3", "4"]:
        recognizer.visualize_features(display)  # Default'''

    start_display()


if __name__ == "__main__":
    main()

