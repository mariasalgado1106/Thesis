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
    groups = process_planner.group_by_tads()

    # Let's test the first valid machining axis found in the model
    test_axes = [axis for axis in groups.keys() if axis != "INACCESSIBLE"]

    if test_axes:
        for axis in test_axes:
            print(f"\n>>> Testing Workholding for Setup Axis: {axis}")
            features_in_setup = groups[axis]

            # CALL YOUR NEW VALIDATION FUNCTION
            # This triggers: Area Ratio, Triangle Selection, and Stability Score
            plf_trio, slf, tlf = process_planner.validate_workholding(axis)

            if plf_trio:
                indices = [f['PLF_idx'] for f in plf_trio]
                print(f"RESULT: Setup {axis} is VALID. PLF Faces: {indices}")
            else:
                print(f"RESULT: Setup {axis} is INVALID or unstable.")
    else:
        print("No accessible machining directions found.")




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