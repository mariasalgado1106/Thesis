from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import analyze_shape
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies

class Process_Plan:
    def __init__(self, my_shape):
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)

        self.recognizer = FeatureRecognition(self.shape)
        self.features = self.recognizer.identify_features()

        self.tad_extractor = TAD_Extraction(self.shape)
        self.dep_extractor = Dependencies(self.shape)
        self.feature_info = self.dep_extractor.identify_relationships()

    def group_by_tads(self):
        groups = {}
        for feat in self.feature_info:
            tads = feat.get('tads', [])
            if not tads:
                if "INACCESSIBLE" not in groups:
                    groups["INACCESSIBLE"] = []
                groups["INACCESSIBLE"].append(feat)
                continue
            for t in tads:
                axis = t['axis']
                if axis not in groups:
                    groups[axis] = []
                if feat not in groups[axis]:
                    groups[axis].append(feat)
        return groups

    def group_by_feat_type(self, features):
        hole_types = ['feat_hole_blind', 'feat_hole_through']
        grouped_types = {'holes': [],
                 'others': []}
        for feat in features:
            if feat['feature_type'] in hole_types:
                grouped_types['holes'].append(feat)
            else:
                grouped_types['others'].append(feat)
        return grouped_types

    def generate_optimized_plan(self):
        # 1. Get features grouped by their possible axes
        groups = self.group_by_tads()

        # 2. Identify the order of setups (e.g., the one with most features first)
        # We sort by the length of the feature list in each group
        sorted_setups = sorted(
            [axis for axis in groups if axis != "INACCESSIBLE"],
            key=lambda x: len(groups[x]),
            reverse=True
        )

        optimized_plan = []
        already_planned = set()

        print("\n" + "=" * 50)
        print("GENERATING OPTIMIZED PROCESS PLAN")
        print("=" * 50)

        for axis in sorted_setups:
            # Only take features that haven't been assigned to a previous setup
            features_to_order = [f for f in groups[axis] if f['feat_idx'] not in already_planned]

            if not features_to_order:
                continue

            print(f"\n>>> Planning Setup: {axis} ({len(features_to_order)} features)")

            # 3. Call your setup_order to sequence the features within this TAD
            ordered_sequence = self.setup_order(features_to_order, axis)

            # Store the result
            optimized_plan.append({
                'setup': axis,
                'sequence': [f['feat_idx'] for f in ordered_sequence]
            })

            # Mark these as done so they aren't produced twice in another setup
            for f in ordered_sequence:
                already_planned.add(f['feat_idx'])

        print("\n" + "=" * 50)
        print("FINAL OPTIMIZED SEQUENCE:")
        for step in optimized_plan:
            print(f"Setup {step['setup']}: {step['sequence']}")
        print("=" * 50 + "\n")

        return optimized_plan

    def setup_order (self, features_of_setup, current_setup_axis):
        # order the features within a setup, based on:
        # 1. feature type (holes and others -> minimize tool swaps)
        # 2. base face area (max 1st for stability)
        # 3. dependencies

        # prioritize starting with "others" and holes for last
        for f in features_of_setup:
            relevant_tad = next((t for t in f['tads'] if t['axis'] == current_setup_axis), None)
            if relevant_tad and relevant_tad['tad_face_index'] != "none":
                # Blind Features -> tad face area
                f['priority_area'] = self.face_data_list[relevant_tad['tad_face_index']]['face_area']
                new = self.recognizer.get_projected_area(f['feat_idx'], current_setup_axis)
                print(f"area inicial = {f['priority_area']} vs nova {new}")
            elif relevant_tad != "none":
                # Through -> max area from all faces
                f['priority_area'] = self.recognizer.get_projected_area(f['feat_idx'], current_setup_axis)

        ordered_list = []
        pending = list(features_of_setup) #feat not done still
        previous_was_hole = 0

        while pending:
            available = [] #available to be considered and produced now
            # available = no dependencies in pending (none at all, or they were already produced)
            for f in pending:
                current_tad = next((t for t in f['tads'] if t['axis'] == current_setup_axis), None)
                axis_deps = current_tad['dependency'] if current_tad else []
                if all(d in [o['feat_idx'] for o in ordered_list] for d in axis_deps):
                    available.append(f)
                    # if dependencies were already ordered, it is available; or if there are none

            if not available:
                # If nothing is available but pending is not empty, we get stuck in a loop
                if pending:
                    ordered_list.extend(pending)
                break

            # 1. prioritized type = others
            grouped = self.group_by_feat_type(available) #groups only the available ones
            if grouped['others'] and previous_was_hole != 1:
                # 2. priority = max area
                chosen = max(grouped['others'], key=lambda x: x['priority_area'])
                previous_was_hole = 0
            elif grouped['holes']:
                # if no other features are ready, take the hole with max area
                chosen = max(grouped['holes'], key=lambda x: x['priority_area'])
                previous_was_hole = 1
            else:
                chosen = max(grouped['others'], key=lambda x: x['priority_area'])
                previous_was_hole = 0

            ordered_list.append(chosen)
            pending.remove(chosen)

        return ordered_list

    def validate_workholding (self, features_of_setup):
        # here i want to see if this setup with these features is possible
        # apply 321 technique based on the final product after this setup
        return None

    def print_grouped_tads_and_dependencies(self):
        groups = self.group_by_tads()

        print("\n" + "=" * 80)
        print(f"{'SETUP (TAD)':<15} | {'FEATURE IDs':<30} | {'DEPENDENCIES'}")
        print("-" * 80)

        for axis in sorted(groups.keys()):
            feats = groups[axis]
            feat_ids = [str(f['feat_idx']) for f in feats]

            for f in feats:
                deps = ", ".join(map(str, f['dependency'])) if f['dependency'] else "None"
                # We print one row per feature in the setup group
                print(f"{axis:<15} | Feature {f['feat_idx']:<22} | Needs: {deps}")

            print("-" * 80)