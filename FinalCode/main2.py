from OCC.Display.SimpleGui import init_display
import os

from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import load_step_file
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies
from SetupPlanning.Setup_Plan import Setup_Plan


def main():
    # 1. Load STEP file
    step_file = os.path.join("STEPFiles", "Part3.stp")
    my_shape = load_step_file(step_file)
    if not my_shape:
        print("Failed to load shape.")
        return

    # 2. Feature Recognition & TAD Extraction
    print("\n" + "=" * 30 + "\nFEATURE RECOGNITION\n" + "=" * 30)
    recognizer = FeatureRecognition(my_shape)
    features = recognizer.identify_features()

    extractor = TAD_Extraction(my_shape)
    extractor.print_tad_table()

    # 3. Process Planning & Workholding Validation
    print("\n" + "=" * 30 + "\nWORKHOLDING VALIDATION\n" + "=" * 30)
    process_planner = Setup_Plan(my_shape)

    '''# --- TEST GRID GENERATION ---
    test_axis = '-z'
    # Manually extract PLFs for the test axis to feed the grid generator
    stock_faces = process_planner.define_stock_faces_list()
    test_plfs = []
    for sf in stock_faces:
        if sf['opposite_TAD'] == test_axis:
            test_plfs.append({
                'PLF_idx': sf['stock_face_idx'],
                'PLF_center': process_planner.face_data_list[sf['stock_face_idx']]['face_center']
            })

    if test_plfs:
        print(f"Generating grid for {test_axis}...")
        grid_points = process_planner.generate_locating_grid(test_plfs, test_axis)

        if len(grid_points) >= 3:
            # 1. Find the locators
            locators, balanced = process_planner.find_PLF_locators(grid_points, test_axis)

            # 2. Get CoG for visualization
            cog = process_planner.get_part_cog()

            print(f"Generated {len(grid_points)} valid grid points.")
            print(f"Locators found: {locators}")
            print(f"Balanced: {balanced}")

            # 3. Visualize everything
            process_planner.visualize_setup_results(grid_points, locators, cog)
        else:
            print("Not enough grid points to find locators.")'''


    # 4. Visualization
    print("\n" + "=" * 30 + "\nVISUALIZATION\n" + "=" * 30)
    choice = input("Visualize Features? [y/n]: ").lower()

    if choice == 'y':
        recognizer.visualize_features_3d(
            show_mesh=True,
            show_face_centers=False,
            show_edges=True,
            show_feat_idx=True,
            show_all_face_centers = True
        )


if __name__ == "__main__":
    main()