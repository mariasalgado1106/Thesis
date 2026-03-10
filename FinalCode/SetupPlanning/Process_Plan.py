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

    def group_by_feat_type(self):
        types = {}
        #separate holes from others
        return types

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

    def generate_optimized_plan(self):
        # here i want to generate a
        return None

    def setup_order (self, features_of_setup, current_setup_axis):
        # order the features within a setup, based on:
        # 1. base face area (max 1st for stability)
        # 2. feature type (holes and others -> minimize tool swaps)
        # 3. dependencies

        # 1. Get area of base face of this setup
        for f in features_of_setup:
            # only the base face area relevant to this setup axis
            relevant_tad = next((t for t in f['tads'] if t['axis'] == current_setup_axis), None)

            if relevant_tad and relevant_tad['tad_face_index'] != "none":
                face_idx = relevant_tad['tad_face_index']
                f['priority_area'] = self.face_data_list[face_idx]['face_area']
            else:
                # Through features or features without a defined base for this TAD
                f['priority_area'] = 0

        sorted_list = []
        pending = list(features_of_setup) #to see which features were still not ordered after

        while pending:
            # Identify features whose dependencies are ALREADY in sorted_list (or have no dependencies)
            available = []
            for f in pending:
                # Check if all dependencies of f are already handled(not in the pending list anymore)
                deps_still_waiting = [d for d in f['dependency'] if any(p['feat_idx'] == d for p in pending)]
                if not deps_still_waiting:
                    available.append(f)

            if not available:
                # Emergency break for circular dependencies to avoid infinite loop
                if pending:
                    sorted_list.extend(pending)
                break

            # 2. Separate by tool groups (Others vs Holes)
            hole_types = ['feat_hole_blind', 'feat_hole_through']
            other_feats = [f for f in available if f['feature_type'] not in hole_types]
            hole_feats = [f for f in available if f['feature_type'] in hole_types]

            # 3. Decision Logic:
            if other_feats:
                # do the one with the biggest area first
                chosen = max(other_feats, key=lambda x: x['priority_area'])
            else:
                # If only holes are ready, start processing them
                # Sorting holes by area here too, in case one is a large counterbore
                chosen = max(hole_feats, key=lambda x: x['priority_area'])

            sorted_list.append(chosen)
            pending.remove(chosen)

        return sorted_list

    def print_setup_sequencing_debug(self):
        """
        Prints a detailed report of how features are ordered within each setup
        based on area, type, and dependency constraints.
        """
        groups = self.group_by_tads()

        print("\n" + "!" * 90)
        print("DEBUG: INTRA-SETUP SEQUENCING REPORT")
        print("!" * 90)

        for axis in sorted(groups.keys()):
            if axis == "INACCESSIBLE": continue

            # Run your ordering logic
            ordered_feats = self.setup_order(groups[axis], axis)

            print(f"\n>>> SETUP AXIS: {axis} ({len(ordered_feats)} features)")
            print(f"{'Order':<6} | {'ID':<4} | {'Type':<20} | {'Base Area':<12} | {'Dependencies'}")
            print("-" * 90)

            for i, f in enumerate(ordered_feats, 1):
                f_idx = f['feat_idx']
                f_type = f['feature_type']
                area = round(f.get('priority_area', 0), 2)
                deps = ", ".join(map(str, f['dependency'])) if f['dependency'] else "None"

                # Tagging the type group for visual clarity
                type_tag = "[HOLE]" if f_type in ['feat_hole_blind', 'feat_hole_through'] else "[MILL]"

                print(f"{i:<6} | {f_idx:<4} | {type_tag} {f_type:<14} | {area:<12} | {deps}")

        print("\n" + "!" * 90 + "\n")

    def validate_workholding (self, features_of_setup):
        # here i want to see if this setup with these features is possible
        # apply 321 technique based on the final product after this setup
        return None