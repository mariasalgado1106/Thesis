from OCC.Display.SimpleGui import init_display
import os

from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import load_step_file
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies
from SetupPlanning.Setup_Plan import Setup_Plan
from SetupPlanning.Workholding import Workholding


def main():
    # 1. Load STEP file
    step_file = os.path.join("STEPFiles", "Part2.stp")
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
    #optimized_plan = process_planner.generate_optimized_plan()

    workholding = Workholding(my_shape)

    '''
    # --- TEST 3-2-1 CONFIGURATION ---
    test_axis = 'y'

    # 1. Run the full validation to get PLF, SLF, and TLF locators
    print(f"Calculating full 3-2-1 setup for {test_axis}...")
    PLF_res, SLF_res, TLF_res, validated = process_planner.validate_workholding(test_axis)

    if validated:
        # Extract coordinates from the results
        p_locs = PLF_res['PLF_locators']
        s_locs = SLF_res['SLF_locators']
        t_locs = TLF_res['TLF_locators']
        cog_point = process_planner.get_part_cog()

    else:
        print("Setup could not be validated.")#'''

    '''
    #TEST WORKHOLDING (VIZUALIZE THE COMMON AREA)
    test_axis = 'x'
    opposite_axis = {'z': '-z', '-z': 'z',
                     'x': '-x', '-x': 'x',
                     'y': '-y', '-y': 'y'}
    test_axis_2 = opposite_axis[test_axis]
    common_points, common_area = workholding.common_parallel_area(test_axis, test_axis_2)
    workholding.visualize_common_area(test_axis, test_axis_2, common_points) #'''

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