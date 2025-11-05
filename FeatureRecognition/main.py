import os
from OCC.Display.SimpleGui import init_display

# Import your custom modules
from geometry_analysis import load_step_file, analyze_shape, print_face_analysis_table
from aag_builder import AAGBuilder
from feature_recognition import FeatureRecognizer


def main():
    """Main execution function."""

    # Initialize display
    display, start_display, add_menu, add_function_to_menu = init_display()

    # Load STEP file
    file_path = os.path.join("STEPFiles", "example_thoughhole.stp")
    my_shape = load_step_file(file_path)

    if not my_shape:
        return

    # Display base shape
    display.DisplayShape(my_shape, update=True, transparency=0.8, color="GRAY")

    # PART 1: Geometry Analysis
    print("\n=== PART 1: GEOMETRY ANALYSIS ===")
    all_faces, face_data_list, analyser = analyze_shape(my_shape)
    print_face_analysis_table(all_faces, face_data_list)

    # PART 2: Build AAG
    print("\n=== PART 2: AAG CONSTRUCTION ===")
    aag_builder = AAGBuilder(face_data_list)
    aag_builder.print_graph_summary()

    # Optional: Export graph data for visualization
    graph_data = aag_builder.export_graph_data()
    print(f"AAG contains {len(graph_data['nodes'])} nodes and {len(graph_data['edges'])} edges")

    # PART 3: Feature Recognition
    print("\n=== PART 3: FEATURE RECOGNITION ===")
    recognizer = FeatureRecognizer(face_data_list, aag_builder)
    features = recognizer.recognize_all_features()

    # Print statistics
    stats = recognizer.get_feature_statistics()
    print("\nFeature Statistics:")
    for feature_type, count in stats.items():
        print(f"  {feature_type}: {count}")

    # Visualize results
    print("\n=== VISUALIZATION ===")
    choice = input("Visualize: (1) Features, (2) Edge Types, (3) Both? [1/2/3]: ")

    if choice in ["1", "3"]:
        recognizer.visualize_features(display)

    if choice in ["2", "3"]:
        recognizer.visualize_edge_types(display)

    if choice not in ["1", "2", "3"]:
        recognizer.visualize_features(display)  # Default

    # Start the display
    start_display()


if __name__ == "__main__":
    main()
