import os
from OCC.Display.SimpleGui import init_display
from FeatureRecognition.geometry_analysis import (load_step_file, analyze_shape, print_face_analysis_table, print_edge_analysis_table)
from FeatureRecognition.aag_builder import AAGBuilder_2D, AAGBuilder_3D
from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.part_visualizer_occ import PartVisualizer_occ
from FeatureRecognition.part_vizualizer_plotly import Part_Visualizer


def main():
    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Load STEP file
    my_shape = load_step_file(os.path.join("STEPFiles", "example4.stp"))
    if not my_shape:
        return

    # PART 1: Geometry Analysis
    print("\n PART 1: GEOMETRY ANALYSIS")
    all_faces, face_data_list, analyser, all_edges, edge_data_list = analyze_shape(my_shape)
    print_face_analysis_table(all_faces, face_data_list)
    print_edge_analysis_table(all_edges, edge_data_list)

    # PART 2: Build AAG
    print("\n PART 2: AAG CONSTRUCTION")
    builder2D = AAGBuilder_2D(my_shape)
    subgraphs_info = builder2D.analyse_subgraphs()

    builder3D = AAGBuilder_3D(my_shape)
    builder3D.load_shape()

    # PART 3: Feature Recognition
    print("\nPART 3: FEATURE RECOGNITION")
    recognizer = FeatureRecognition(my_shape)
    features = recognizer.identify_features()

    # stats
    print(f"Detected {len(features)} features")
    for f in features:
        print(f["feature_type"], f["node_indices"])

    # Visualize results
    print("\n VISUALIZATION")
    choice = input(
        "Visualize: "
        "(0) Numbered Faces,"
        "(1) Edge Types, "
        "(2) Features, "
        "(3) Both, "
        "(4) 2D AAG + Subgraphs, "
        "(5) 3D AAG, "
        "(6) 3D AAG without convex edges "
        "[0/1/2/3/4/5/6]: "
    )

    viz_occ = PartVisualizer_occ(builder2D, recognizer)
    viz = Part_Visualizer(builder3D)

    if choice == "0":
        #builder3D.visualize_numbered_faces()
        viz.visualize_numbered_faces()

    if choice in ["1"]:
        viz.visualize_geometric_edges()

    if choice in ["2"]: #dont show faces or edges
        recognizer.visualize_features_3d(
            show_mesh=True,
            show_face_centers=False,
            show_edges=False
        )

    if choice in ["3"]: #show faces & edges
        recognizer.visualize_features_3d(
            show_mesh=True,
            show_face_centers=True,
            show_edges=True
        )

    if choice == "4":
        builder2D.visualize_2d_aag()

    if choice == "5":
        builder3D.visualize_3d_aag(hide_convex=False)

    if choice == "6":
        builder3D.visualize_3d_aag(hide_convex=True)

    start_display()


if __name__ == "__main__":
    main()

