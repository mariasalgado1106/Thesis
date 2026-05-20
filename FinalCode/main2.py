import os

from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import load_step_file
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies
from SetupPlanning.Setup_Plan import Setup_Plan
from SetupPlanning.Workholding import Workholding


def main():
    # 1. Load STEP file
    step_file = os.path.join("STEPFiles", "Part4.stp")
    my_shape = load_step_file(step_file)
    if not my_shape:
        print("Failed to load shape.")
        return

    # 2. Feature Recognition & TAD Extraction
    print("\n" + "=" * 30 + "\nFEATURE RECOGNITION\n" + "=" * 30)
    recognizer = FeatureRecognition(my_shape)
    features = recognizer.identify_features()

    extractor = TAD_Extraction(my_shape, recognizer=recognizer)
    extractor.print_tad_table()

    dependencies = Dependencies(my_shape, recognizer=recognizer)
    dependencies.print_dependency_table()

    # 3. Process Planning & Workholding Validation
    print("\n" + "=" * 30 + "\nWORKHOLDING VALIDATION\n" + "=" * 30)
    process_planner = Setup_Plan(my_shape)
    workholding = Workholding(my_shape, recognizer)
    optimized_plan = workholding.optimized_plan
    workholding.clamping_faces()

    # 4. Visualization
    print("\n" + "=" * 30 + "\nVISUALIZATION\n" + "=" * 30)
    choice = input(
        "Visualize:"
        "(0) Only Features,"
        "(1) Locators ").lower()

    if choice == '0':
        recognizer.visualize_features_3d(
            show_mesh=True,
            show_face_centers=False,
            show_edges=True,
            show_feat_idx=True,
            show_all_face_centers = True
        )

    if choice == '1':
        process_planner.visualize_all_setups_3d(optimized_plan)


if __name__ == "__main__":
    main()