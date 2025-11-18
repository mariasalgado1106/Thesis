import os
from OCC.Display.SimpleGui import init_display
from geometry_analysis import (load_step_file, analyze_shape, print_face_analysis_table)
from aag_builder import AAGBuilder
from feature_recognition import FeatureRecognizer


def main():
    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Load STEP file
    my_shape = load_step_file(os.path.join("STEPFiles", "Part3.stp"))

    if not my_shape:
        return

    # Display base shape
    display.DisplayShape(my_shape, update=True, transparency=0.8)

    # PART 1: Geometry Analysis
    print("\n PART 1: GEOMETRY ANALYSIS")
    all_faces, face_data_list, analyser = analyze_shape(my_shape)
    print_face_analysis_table(all_faces, face_data_list)


    # PART 2: Build AAG
    print("\n PART 2: AAG CONSTRUCTION")
    builder = AAGBuilder(my_shape)

    # PART 3: Feature Recognition
    '''print("\n PART 3: FEATURE RECOGNITION")
    recognizer = FeatureRecognizer(face_data_list, analyse_subgraphs)
    features = recognizer.recognize_all_features()

    # Print statistics
    stats = recognizer.get_feature_statistics()
    print("\nFeature Statistics:")
    for feature_type, count in stats.items():
        print(f"  {feature_type}: {count}")'''

    # Visualize results
    print("\n VISUALIZATION")
    choice = input("Visualize: (1) Features, (2) Edge Types, (3) Both, (4) AAG + Subgraphs [1/2/3/4]: ")

    '''if choice in ["1", "3"]:
        recognizer.visualize_features(display)

    if choice in ["2", "3"]:
        recognizer.visualize_edge_types(display)'''

    if choice == "4":
        builder.visualize_aag()

    '''if choice not in ["1", "2", "3", "4"]:
        recognizer.visualize_features(display)  # Default'''

    start_display()


if __name__ == "__main__":
    main()

