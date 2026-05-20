from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import analyze_shape
import numpy as np
from OCC.Core.BRepIntCurveSurface import BRepIntCurveSurface_Inter
from OCC.Core.gp import gp_Lin, gp_Pnt, gp_Dir

class TAD_Extraction:
    def __init__(self, my_shape, recognizer=None):
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)
        self.recognizer = recognizer if recognizer else FeatureRecognition(self.shape)
        self.features = self.recognizer.identify_features()

    def validate_tad(self, start_coords, candidate_tad_vec):
        pnt = gp_Pnt(start_coords[0], start_coords[1], start_coords[2])
        clearance_dir = gp_Dir(-candidate_tad_vec[0], -candidate_tad_vec[1], -candidate_tad_vec[2])
        ray = gp_Lin(pnt, clearance_dir)
        intersector = BRepIntCurveSurface_Inter()
        intersector.Init(self.shape, ray, 1e-4)
        while intersector.More():
            if intersector.W() > 1e-3:
                w_dist = intersector.W()
                hit_pnt = intersector.Pnt()
                print(f"  [BUMP] Hit at distance W={w_dist:.4f}")
                print(f"  Start: ({start_coords[0]:.2f}, {start_coords[1]:.2f}, {start_coords[2]:.2f})")
                print(f"  Impact: ({hit_pnt.X():.2f}, {hit_pnt.Y():.2f}, {hit_pnt.Z():.2f})")
                return False #bump
            intersector.Next()
        return True #if there is no intersection

    def get_tads(self):
        feature_tads = []
        for feature in self.features:
            feat_idx = feature['feat_idx']
            feat_type = feature['feature_type']
            feat_faces = feature['node_indices']
            tad_faces = feature['tad_faces']
            tads_for_this_feature = []
            # 1. features with base faces (Blind)
            if tad_faces:
                for tad_face in tad_faces:
                    tad_face_data = self.face_data_list[tad_face]
                    feature_anchor = self.face_data_list[tad_face]['face_center']
                    if tad_face_data['type'] == "Plane": raw_coords = tad_face_data['normal_vector_coords']
                    elif tad_face_data['type'] == "Cylinder": raw_coords = tad_face_data['cylinder_axis_coords']
                    else: continue
                    # Invert to point INWARD
                    candidate_vec = [-raw_coords[0], -raw_coords[1], -raw_coords[2]]
                    if self.validate_tad(feature_anchor, candidate_vec):
                        tads_for_this_feature.append({
                            'tad_face_index': tad_face,
                            'origin' : feature_anchor,
                            'vector_coords': candidate_vec,
                            'axis': self.get_axis_label(candidate_vec)})
                    else: #TRY OTHER DIRECTION
                        opposite_vec = [raw_coords[0], raw_coords[1], raw_coords[2]]
                        if self.validate_tad(feature_anchor, opposite_vec):
                            tads_for_this_feature.append({
                                'tad_face_index': tad_face,
                                'origin': feature_anchor,
                                'vector_coords': opposite_vec,
                                'axis': self.get_axis_label(opposite_vec)})
                        else:
                            print(f"feature {feat_idx} has invalid tad: {tad_face}")
            # 2. Through Features -> no base
            else:
                all_wall_pnts = [np.array(self.face_data_list[f]['face_center']) for f in feat_faces]
                feature_anchor = np.mean(all_wall_pnts, axis=0).tolist()
                first_wall_data = self.face_data_list[feat_faces[0]]
                if first_wall_data['type'] == "Cylinder": vec = first_wall_data['cylinder_axis_coords']
                else: vec = self.calculate_through_pocket_axis(feat_faces)
                if vec:#both directions originating from the same center
                    for sign in [1, -1]:
                        v = [sign * c for c in vec]
                        tads_for_this_feature.append({
                            'tad_face_index': "none",
                            'origin': feature_anchor,
                            'vector_coords': v,
                            'axis': self.get_axis_label(v)})
            feature_tads.append({
                'feat_idx': feat_idx,
                'feature_type': feat_type,
                'all_faces': feat_faces,
                'tad_base_faces': tad_faces,
                'tads': tads_for_this_feature})
        return feature_tads

    def get_axis_label(self, coords, tol=1e-3):
        if not coords: return "No axis"
        nx, ny, nz = coords
        if abs(nx - 1.0) < tol: return "x"
        if abs(nx + 1.0) < tol: return "-x"
        if abs(ny - 1.0) < tol: return "y"
        if abs(ny + 1.0) < tol: return "-y"
        if abs(nz - 1.0) < tol: return "z"
        if abs(nz + 1.0) < tol: return "-z"
        return "Other"

    def calculate_through_pocket_axis(self, feat_faces):
        if len(feat_faces) < 2: return None
        n1 = np.array(self.face_data_list[feat_faces[0]]['normal_vector_coords'])
        for i in range(1, len(feat_faces)):
            n2 = np.array(self.face_data_list[feat_faces[i]]['normal_vector_coords'])
            axis = np.cross(n1, n2)
            mag = np.linalg.norm(axis)
            if mag > 1e-6:  # Ensure they aren't parallel walls
                normalized_axis = (axis / mag).tolist()
                return normalized_axis
        return None

    def print_tad_table(self):
        tads_data = self.get_tads()
        print("\n" + "=" * 95)
        print(f"{'ID':<4} | {'Feature Type':<22} | {'TAD Faces':<12} | {'TAD Axis':<10} | {'All Faces'}")
        print("-" * 95)
        for feat in tads_data:
            f_idx = feat['feat_idx']
            f_type = feat['feature_type']
            base_faces_str = ", ".join(map(str, feat['tad_base_faces'])) if feat['tad_base_faces'] else "N/A"
            if feat['tads']: tad_axes = ", ".join(set([t['axis'] for t in feat['tads']]))
            else: tad_axes = "Through"
            all_faces_str = ", ".join(map(str, feat['all_faces']))
            print(f"{f_idx:<4} | {f_type:<22} | {base_faces_str:<12} | {tad_axes:<10} | {all_faces_str}")
        print("=" * 95 + "\n")

class Dependencies:
    def __init__(self, my_shape, recognizer=None):
        self.shape = my_shape
        self.recognizer = recognizer if recognizer else FeatureRecognition(self.shape)
        self.tad_extractor = TAD_Extraction(self.shape, recognizer=self.recognizer)
        self.feature_info = self.tad_extractor.get_tads()
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)
        for feat in self.feature_info:
            for t in feat.get('tads', []): t['dependency'] = []

    def identify_relationships(self):
        defined_relationships = []
        # 1. BLIND FEATURE IN FRONT OF THROUGH FEATURE WITH THE SAME TAD
        fixed_types = ['feat_hole_blind', 'feat_pocket_blind']
        flexible_types = ['feat_hole_through', 'feat_pocket_through']
        for f_flex in self.feature_info:
            if f_flex['feature_type'] in flexible_types:
                for f_fixed in self.feature_info:
                    if f_fixed['feature_type'] in fixed_types:
                        # Check if relationship already exists
                        if (f_flex, f_fixed) in defined_relationships:
                            continue
                        # Combined Vector + Collinearity check
                        is_aligned, match_vec, match_axis = self.check_alignment(f_flex, f_fixed)
                        if is_aligned:
                            for t in f_flex['tads']:
                                if t['axis'] == match_axis:
                                    if f_fixed['feat_idx'] not in t['dependency']:
                                        t['dependency'].append(f_fixed['feat_idx'])
                            defined_relationships.append((f_flex, f_fixed))
        # 2. FEATURE IN FRONT OF FEATURE with same center vector
        for f1 in self.feature_info:
            for f2 in self.feature_info:
                if f1 == f2 or (f1, f2) in defined_relationships: continue
                is_aligned, match_vec, match_axis = self.check_alignment(f1, f2)
                if is_aligned:
                    f_front, f_back = self.compare_coordinates(match_axis, f1, f2)
                    if f_front and f_back:
                        for t in f_back['tads']:
                            if t['axis'] == match_axis:
                                if f_front['feat_idx'] not in t['dependency']:
                                    t['dependency'].append(f_front['feat_idx'])
                        defined_relationships.append((f1, f2))
        # 3. FEATURE IN FRONT OF FEATURE with DIF center vector
        for f1 in self.feature_info:
            for f2 in self.feature_info:
                if f1 == f2 or (f1, f2) in defined_relationships: continue
                has_shared_setup, match_axis = self.check_shared_setup(f1, f2)
                if has_shared_setup:
                    f_front, f_back = self.compare_coordinates(match_axis, f1, f2)
                    if f_front and f_back:
                        front_base_indices = f_front.get('tad_base_faces', [])
                        back_base_indices = f_back.get('tad_base_faces', [])
                        back_wall_indices = [idx for idx in f_back['all_faces'] if idx not in back_base_indices]
                        have_relationship = False
                        for b_idx in front_base_indices:
                            convex_neighbors = self.face_data_list[b_idx].get('convex_adjacent', [])
                            if any(neighbor in back_wall_indices for neighbor in convex_neighbors):
                                have_relationship = True
                                break
                        if have_relationship:
                            for t in f_back['tads']:
                                if t['axis'] == match_axis:
                                    if f_front['feat_idx'] not in t['dependency']:
                                        t['dependency'].append(f_front['feat_idx'])
                            defined_relationships.append((f1, f2))
        return self.feature_info

    def check_alignment(self, f_flex, f_fixed, vec_tol=1e-5, col_tol=1.0):
        for t_flex in f_flex.get('tads', []):
            v_flex = np.array(t_flex['vector_coords'])
            p_flex = np.array(t_flex['origin'])
            for t_fixed in f_fixed.get('tads', []):
                v_fixed = np.array(t_fixed['vector_coords'])
                p_fixed = np.array(t_fixed['origin'])
                # Directional Check
                if np.allclose(v_flex, v_fixed, atol=vec_tol):
                    # Collinearity
                    dist_vec = p_fixed - p_flex
                    cross_prod = np.cross(dist_vec, v_flex)
                    perpendicular_error = np.linalg.norm(cross_prod)
                    if perpendicular_error < col_tol: return True, v_flex, t_flex['axis']
        return False, None, None

    def compare_coordinates(self, match_axis, f1_dict, f2_dict):
        p1 = f1_dict['tads'][0]['origin']
        p2 = f2_dict['tads'][0]['origin']
        # Determine which feature is "in front" based on the axis label
        if match_axis == "-z":
            if p1[2] > p2[2]: return f1_dict, f2_dict  # f1 in front of f2
            return f2_dict, f1_dict
        elif match_axis == "z":
            if p1[2] < p2[2]: return f1_dict, f2_dict
            return f2_dict, f1_dict
        elif match_axis == "-x":
            if p1[0] > p2[0]: return f1_dict, f2_dict
            return f2_dict, f1_dict
        elif match_axis == "x":
            if p1[0] < p2[0]: return f1_dict, f2_dict
            return f2_dict, f1_dict
        elif match_axis == "-y":
            if p1[1] > p2[1]: return f1_dict, f2_dict
            return f2_dict, f1_dict
        elif match_axis == "y":
            if p1[1] < p2[1]: return f1_dict, f2_dict
            return f2_dict, f1_dict
        return None, None

    def check_shared_setup(self, f1, f2, vec_tol=1e-5):
        for t1 in f1.get('tads', []):
            v1 = np.array(t1['vector_coords'])
            for t2 in f2.get('tads', []):
                v2 = np.array(t2['vector_coords'])
                if np.allclose(v1, v2, atol=vec_tol): return True, t1['axis']
        return False, None

    def print_dependency_table(self):
        data = self.identify_relationships()
        print("\n" + "=" * 105)
        print(f"{'ID':<4} | {'Feature Type':<25} | {'Role':<12} | {'TADs':<10} | {'Setup-Specific Dependencies'}")
        print("-" * 105)
        fixed_types = ['feat_hole_blind', 'feat_pocket_blind']
        for feat in data:
            f_idx = feat['feat_idx']
            f_type = feat['feature_type']
            if f_type in fixed_types: role = "FIXED"
            elif "through" in f_type: role = "FLEXIBLE"
            else: role = "OTHER"
            available_axes = [t['axis'] for t in feat.get('tads', [])]
            tad_str = ", ".join(available_axes) if available_axes else "NONE"
            dep_list = []
            for t in feat.get('tads', []):
                for dep_id in t.get('dependency', []): dep_list.append(f"{dep_id} ({t['axis']})")
            dep_str = ", ".join(dep_list) if dep_list else "---"
            print(f"{f_idx:<4} | {f_type:<25} | {role:<12} | {tad_str:<10} | {dep_str}")
        print("=" * 105 + "\n")