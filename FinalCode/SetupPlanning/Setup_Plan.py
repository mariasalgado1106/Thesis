from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import analyze_shape, get_stock_box
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies

import numpy as np
import itertools

class Setup_Plan:
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

    def define_stock_faces_list(self):
        # stock faces list -> to after use for workholding
        stock_faces = []
        for f in self.face_data_list:
            is_stock_face = f['stock_face']
            f_idx = f['index']
            f_area = f['face_area']
            opposite_tad = f['normal_vector_axis']

            if is_stock_face == "Yes":
                stock_faces.append({
                    'stock_face_idx': f_idx,
                    'area': f_area,
                    'opposite_TAD': opposite_tad, # basically, if z then this can be base face for TAD z
                    'perpendicular_stock_faces': []
                })

        for sf in stock_faces:
            for sf_2 in stock_faces:
                # Check if they are different and sf_2 is not already added
                if sf['stock_face_idx'] != sf_2['stock_face_idx']:
                    if self.are_faces_perpendicular(sf['stock_face_idx'], sf_2['stock_face_idx']):
                        if sf_2['stock_face_idx'] not in sf['perpendicular_stock_faces']:
                            sf['perpendicular_stock_faces'].append(sf_2['stock_face_idx'])
                            sf_2['perpendicular_stock_faces'].append(sf['stock_face_idx'])

        return stock_faces

    def are_faces_perpendicular(self, f1_idx, f2_idx, tolerance=1e-3):
        normal1 = self.face_data_list[f1_idx]['normal_vector_coords']
        normal2 = self.face_data_list[f2_idx]['normal_vector_coords']

        n1 = np.array(normal1)
        n2 = np.array(normal2)
        mag1 = np.linalg.norm(n1)
        mag2 = np.linalg.norm(n2)
        #avoid division by 0
        if mag1 < 1e-9 or mag2 < 1e-9:
            return False
        # Normalize
        n1_unit = n1 / mag1
        n2_unit = n2 / mag2

        dot_product = np.dot(n1_unit, n2_unit)

        # If dot product is ~0, they are 90 degrees apart
        return abs(dot_product) < tolerance

    def get_original_stock_area(self, axis):
        xmin, ymin, zmin, xmax, ymax, zmax, _ = get_stock_box(self.shape)

        dx = abs(xmax - xmin)
        dy = abs(ymax - ymin)
        dz = abs(zmax - zmin)

        # Area depends on which plane we are looking at
        if 'x' in axis:
            return dy * dz
        elif 'y' in axis:
            return dx * dz
        elif 'z' in axis:
            return dx * dy
        return 0

    def find_best_3_locators_for_trio(self, PLFs, axis):
        # prepare data
        axis_map = {'z': (0, 1), '-z': (0, 1), 'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)}
        dims = axis_map.get(axis.lower(), (0, 1)) #the 2 necessary axis (if axis =-z, then its (0,1) which is x and y
        _, _, _, _, _, _, cog_pnt = get_stock_box(self.shape) # center of gravity
        cog = np.array([cog_pnt.X(), cog_pnt.Y(), cog_pnt.Z()])

        best_score = -1
        best_trio = None
        # Iterate through every combination of 3 faces (trio)
        for trio in itertools.combinations(PLFs, 3):
            p1 = np.array(trio[0]['PLF_center'])
            p2 = np.array(trio[1]['PLF_center'])
            p3 = np.array(trio[2]['PLF_center'])
            # Calculate Triangle Area (Geometric Stability)
            tri_area = self.calculate_2d_area(p1, p2, p3, dims)
            # Check if CoG is inside (Balance)
            is_balanced = self.is_point_in_triangle(cog, p1, p2, p3, dims)
            balance_multiplier = 1 if is_balanced else 0.1  # Heavily reward balanced trios

            # Total Face Area (Mechanical Support)
            support_area = trio[0]['PLF_area'] + trio[1]['PLF_area'] + trio[2]['PLF_area']

            # FINAL SCORE
            # We want a large triangle on large faces that contains the CoG
            score = tri_area * support_area * balance_multiplier

            if score > best_score:
                best_score = score
                best_trio = trio

        return best_trio, best_score

    def calculate_2d_area(self, p1, p2, p3, dims):
        x1, y1 = p1[dims[0]], p1[dims[1]]
        x2, y2 = p2[dims[0]], p2[dims[1]]
        x3, y3 = p3[dims[0]], p3[dims[1]]
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    def is_point_in_triangle(self, p, a, b, c, dims):
        A = self.calculate_2d_area(a,b,c,dims)
        A1 = self.calculate_2d_area(p,b,c,dims)
        A2 = self.calculate_2d_area(a,p,c,dims)
        A3 = self.calculate_2d_area(a,b,p,dims)

        return abs(A - (A1 + A2 + A3)) < 1e-3

    def validate_workholding (self, axis):
        # apply 321 technique based on the final part (worst case scenario)
        PLFs = []
        PLF = None
        SLF = None
        TLF = None

        ##########################################################
        # 1. Get stock faces and possible base face (Primary Locating Face)
        stock_faces = self.define_stock_faces_list()
        PLF_total_area = 0
        for stock_face in stock_faces:
            sf_axis = stock_face['opposite_TAD']
            sf_idx = stock_face['stock_face_idx']
            sf_center = self.face_data_list[sf_idx]['face_center']
            sf_area = stock_face['area']
            if sf_axis == axis:
                PLFs.append({ #list of the coplanar faces in that plane
                    'PLF_idx': sf_idx,
                    'PLF_center': sf_center,
                    'PLF_area': sf_area
                    })
                PLF_total_area = PLF_total_area + sf_area
                print(f"Face {stock_face['stock_face_idx']} is a candidate for PLF of Setup of TAD {axis}")

        ###############################################################
        # 2. Validate these PLFs and get 3 points
        validated = True
        # 2.1. Validate based on area ratio (30% of orginal)
        original_area = self.get_original_stock_area(axis)
        if original_area > 0:
            area_ratio = (PLF_total_area / original_area) * 100
            print(f"Total PLF Area: {PLF_total_area:.2f} / Original: {original_area:.2f} ({area_ratio:.1f}%)")
            if area_ratio < 30:
                print(f"WARNING: Stability compromised! Only {area_ratio:.1f}% of the base remains.")
                validated = False

        # 2.2 Identify 3 specific locator points based on centers
        nr_PLFs = len(PLFs)
        if validated and nr_PLFs >= 3:
            PLF, score = self.find_best_3_locators_for_trio(PLFs,axis)
            if PLF:
                print(f"SUCCESS: Best Trio found with Stability Score: {score:.2f}")
                print(f"Selected Faces: {[f['PLF_idx'] for f in PLF]}")
        '''elif validated and nr_PLFs == 2:
            PLF, score = self.find_best_3_locators_for_duo(PLFs,axis)'''

        return PLF, SLF, TLF









    def generate_optimized_plan(self):
        # 1. features grouped by tads
        groups = self.group_by_tads()

        # 2. order tads (max features 1st)
        # sort by the length of the feature list in each group
        sorted_setups = sorted(
            [axis for axis in groups if axis != "INACCESSIBLE"],
            key=lambda x: len(groups[x]),
            reverse=True
        )

        optimized_plan = [] # setups in order, respective features, PLF, SLF, TLF
        already_planned = set() #features that were already done, fol filtering after

        print("\n" + "=" * 50)
        print("GENERATING OPTIMIZED PROCESS PLAN")
        print("=" * 50)

        for axis in sorted_setups:
            # only features that haven't been assigned to a previous setup
            features_to_order = [f for f in groups[axis] if f['feat_idx'] not in already_planned]

            if not features_to_order:
                continue

            print(f"\n>>> Planning Setup: {axis} ({len(features_to_order)} features)")

            # 3. Validate setup (check the workholding and faces used)
            PLF, SLF, TLF = self.validate_workholding(axis)

            # 4. sequence the features within this TAD (not relevant but it's done)
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