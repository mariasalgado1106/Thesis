from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import analyze_shape, get_stock_box
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies
from SetupPlanning.Setup_Plan import Setup_Plan

import numpy as np


class Workholding:
    def __init__(self, my_shape, recognizer=None):
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)
        (self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax,
         self.stock_box_center) = get_stock_box(self.shape)

        self.recognizer = recognizer if recognizer else FeatureRecognition(self.shape)
        self.features = self.recognizer.identify_features()
        self.colors_rgb = self.recognizer.colors_rgb

        self.setup_plan = Setup_Plan(self.shape, recognizer=self.recognizer)
        self.stock_faces = self.setup_plan.define_stock_faces_list()
        self.optimized_plan = self.setup_plan.generate_optimized_plan()

    # Helper functions
    def generate_grid (self, axis, step_size=0.5):
        axis_map = {'z': (0, 1, 2), '-z': (0, 1, 2),
                    'x': (1, 2, 0), '-x': (1, 2, 0),
                    'y': (0, 2, 1), '-y': (0, 2, 1)}
        opposite_axis ={'z': '-z', '-z': 'z',
                        'x': '-x', '-x': 'x',
                        'y': '-y', '-y': 'y'}
        idx1, idx2, fixed_idx = axis_map[axis]
        bounds = [(self.xmin, self.xmax), (self.ymin, self.ymax), (self.zmin, self.zmax)]
        clamping_height = 5
        dim1_range = np.arange(bounds[idx1][0], bounds[idx1][1], step_size)
        dim2_range = np.arange(bounds[idx2][0], bounds[idx2][1], step_size)
        faces, grid_points = [], []
        opposite_axis_face = opposite_axis[axis]
        for face in self.stock_faces:
            if face['opposite_TAD'] == opposite_axis_face:
                faces.append(face['stock_face_idx'])
        h_val = self.face_data_list[faces[0]]['face_center'][fixed_idx] if faces else 0
        for v1 in dim1_range:
            for v2 in dim2_range:
                in_face = any(self.setup_plan._is_point_in_face_mesh(v1, v2,self.face_data_list[f]['mesh_vertices'],
                                                        self.face_data_list[f]['mesh_triangles'], idx1, idx2)
                               for f in faces)
                if in_face:
                    pnt = [0, 0, 0]
                    pnt[idx1], pnt[idx2], pnt[fixed_idx] = v1, v2, h_val
                    grid_points.append(pnt)

        return grid_points

    def common_parallel_area (self, fa1, fa2, step_size=0.5):
        axis_map = {'z': (0, 1, 2), '-z': (0, 1, 2),
                    'x': (1, 2, 0), '-x': (1, 2, 0),
                    'y': (0, 2, 1), '-y': (0, 2, 1)}
        idx1, idx2, fixed_idx = axis_map[fa1]
        bounds = [(self.xmin, self.xmax), (self.ymin, self.ymax), (self.zmin, self.zmax)]
        dim1_range = np.arange(bounds[idx1][0], bounds[idx1][1], step_size)
        dim2_range = np.arange(bounds[idx2][0], bounds[idx2][1], step_size)
        faces1, faces2, grid_points = [], [], []
        for face in self.stock_faces:
            if face['opposite_TAD'] == fa1:
                faces2.append(face['stock_face_idx'])
            elif face['opposite_TAD'] == fa2:
                faces1.append(face['stock_face_idx'])
        h_val = (self.face_data_list[faces1[0]]['face_center'][fixed_idx] +
                 self.face_data_list[faces2[0]]['face_center'][fixed_idx]) / 2 if (faces1 and faces2) else 0
        for v1 in dim1_range:
            for v2 in dim2_range:
                in_face1 = any(self.setup_plan._is_point_in_face_mesh(v1, v2,self.face_data_list[f]['mesh_vertices'],
                                                        self.face_data_list[f]['mesh_triangles'], idx1, idx2)
                               for f in faces1)
                if in_face1:
                    in_face2 = any(self.setup_plan._is_point_in_face_mesh(v1, v2,self.face_data_list[f]['mesh_vertices'],
                                                        self.face_data_list[f]['mesh_triangles'], idx1, idx2)
                                   for f in faces2)
                    if in_face2:
                        res_pnt = [0.0, 0.0, 0.0]
                        res_pnt[idx1] = v1
                        res_pnt[idx2] = v2
                        res_pnt[fixed_idx] = h_val
                        grid_points.append(tuple(res_pnt))

        total_area = len(grid_points) * (step_size ** 2)
        print(f"Total Common Clamping Area: {total_area} mm²")
        return grid_points, total_area

    def find_height_and_length (self, common_pts, setup, face_axis):
        dual_axis_map = { #input (setup, face axis) -> output(length and height)
            'z': {'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)},
            '-z': {'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)},
            'x': {'y': (2, 0), '-y': (2, 0), 'z': (1, 0), '-z': (1, 0)},
            '-x': {'y': (2, 0), '-y': (2, 0), 'z': (1, 0), '-z': (1, 0)},
            'y': {'x': (2, 1), '-x': (2, 1), 'z': (0, 1), '-z': (0, 1)},
            '-y': {'x': (2, 1), '-x': (2, 1), 'z': (0, 1), '-z': (0, 1)}
        }
        step_size = 0.5
        if not common_pts:
            return 0, 0, 0
        idx_len, idx_height = dual_axis_map[setup][face_axis]
        pts_arr = np.array(common_pts)

        # If positive setup -> the part is flipped
        is_positive_setup = ['x', 'y', 'z']
        if setup in is_positive_setup:
            stock_min_h = [self.xmax, self.ymax, self.zmax][idx_height]
        else:
            stock_min_h = [self.xmin, self.ymin, self.zmin][idx_height]

        # 1. Group points by their "length" coordinate (columns)
        columns = {}
        for p in pts_arr:
            l_coord = round(p[idx_len], 2)
            if l_coord not in columns:
                columns[l_coord] = []
            columns[l_coord].append(p[idx_height])

        # 2. For every column, find its "continuous height" from the bottom
        min_heights, max_heights = [], []
        for l_coord, heights in columns.items():
            if setup in is_positive_setup:
                h_sorted = sorted(heights, reverse=True)
            else:
                h_sorted = sorted(heights)
            h_limit = h_sorted[0]
            max_heights.append(abs(round(stock_min_h - h_sorted[len(h_sorted)-1])))

            for i in range(1, len(h_sorted)):
                if abs(h_sorted[i] - h_sorted[i - 1]) > (step_size * 1.1):
                    break
                else:
                    h_limit = h_sorted[i]
            min_heights.append(abs(round(stock_min_h - h_limit)))
            continue

        # 3. H_min is the MINIMUM without intersecting any features
        # h_max is the MINIMUM max height without intersecting tool (feats in that setup)
        h_min = min(min_heights) if min_heights else 0
        h_max = min(max_heights) if max_heights else 0

        # max_len is the total horizontal span
        max_len = np.max(pts_arr[:, idx_len]) - np.min(pts_arr[:, idx_len])

        return max_len, idx_len, h_min, h_max, idx_height

    # ACTUAL
    def clamping_faces (self):
        perpendicular_axis = {'z': ('x', '-x', 'y', '-y'), '-z': ('x', '-x', 'y', '-y'),
                              'x': ('z', '-z', 'y', '-y'), '-x': ('z', '-z', 'y', '-y'),
                              'y': ('x', '-x', 'z', '-z'), '-y': ('x', '-x', 'z', '-z')}
        clamping_faces_info = []
        print("\n--- ANALYZING CLAMPING OPTIONS PER SETUP ---")
        for setup in self.optimized_plan:
            setup_axis = setup['setup']
            pf1,pf2,pf3,pf4 = perpendicular_axis[setup_axis]
            pairs_parallel_faces = {(pf1,pf2), (pf3,pf4)}
            clamping_pairs = []
            print(f"\nSetup {setup_axis}:")

            for fa1,fa2 in pairs_parallel_faces: #fa = face axis
                print(f"VALIDATING Pair {fa1}/{fa2}.")
                max_min_pts = {'x': (self.xmin, self.xmax),
                               'y': (self.ymin, self.ymax),
                               'z': (self.zmin, self.zmax)}
                # Validate based on clamping area, define max height without interfering with a feature of setup,
                # height of part if clamped in that way vs height of clamp (stability)
                # stability score
                # 1. Is there Clamping Area?
                common_pts, common_area = self.common_parallel_area(fa1, fa2)
                if not common_pts:
                    print(f"  Pairs {fa1}/{fa2}: No common area found.")
                    continue

                # 2. Clamping Width
                clamping_width = abs(max_min_pts[fa1][1] - max_min_pts[fa1][0])

                # 3. Total Height of part vs max Height of clamping area
                max_len, idx_len, h_min, h_max, idx_height = self.find_height_and_length(common_pts, setup_axis, fa1)
                axis_letter = setup_axis.replace('-', '')
                total_part_height = max_min_pts[axis_letter][1] - max_min_pts[axis_letter][0]
                if h_max < 0.4 * total_part_height:  # 40% of total height
                    print(f"Max height of clamping: {h_max},  too small for total height of {total_part_height}.")
                    continue

                # 4. Validate length
                len_axis = {  # input (setup, face axis) -> output(length and height)
                    'z': {'x': 'y', 'y': 'x'},
                    'x': {'y': 'z', 'z': 'y'},
                    'y': {'x': 'z', 'z': 'x'},
                }
                idx_len = len_axis[axis_letter][fa1]
                total_len = abs(max_min_pts[idx_len][1] - max_min_pts[idx_len][0])
                if max_len < 0.5* total_len:
                    print(f"Max length of clamping {max_len} too small.")
                    continue

                # 5. Contact area validation
                is_pos = setup_axis in ['x', 'y', 'z']
                ref_floor = max_min_pts[setup_axis.replace('-', '')][1 if is_pos else 0]
                if is_pos:# If setup is +X, height goes "down" from Xmax
                    filtered_pts = [p for p in common_pts if abs(p[idx_height] - ref_floor) <= h_max]
                else:# If setup is -X, height goes "up" from Xmin
                    filtered_pts = [p for p in common_pts if abs(p[idx_height] - ref_floor) <= h_max]
                common_area_filt = len(filtered_pts) * (0.5 ** 2) #0.5 is step size
                theoretical_jaw_area = max_len * h_max
                contact_ratio = common_area / (theoretical_jaw_area + 1e-6)

                if contact_ratio < 0.4:  # Require at least 40% contact in the grip zone
                    print(f"Contact ratio {contact_ratio:.2f} too low.")
                    continue

                # 6. Stability Score: Favor larger area, larger height coverage, and wider contact
                stability_score = (common_area) * (h_max / total_part_height) * (max_len)

                print(f"Pair of faces {fa1}/{fa2} sucessfully validated.")
                print(f"-> Clamping Width: {clamping_width} mm.")
                print(f"-> Max Heigth without intersecting features of this setup: {h_max} mm.")
                print(f"-> Min Height without intersecting any feature's openings: {h_min} mm.")
                print(f"-> Length of common area: {max_len} mm.")
                print(f"-> Stability score: {stability_score}.")

                clamping_pairs.append({
                        'face_axis': (fa1, fa2),
                        'common_pts': common_pts,
                        'common_area': common_area,
                        'clamping_width': clamping_width,
                        'clamping_max_length': max_len,
                        'clamping_min_height': h_min,
                        'clamping_max_height': h_max,
                        'stability_score': stability_score
                    })
            clamping_faces_info.append({
                'setup_axis': setup_axis,
                'face_pairs': clamping_pairs
            })

        return clamping_faces_info

    def final_clamping_suggestion(self):
        clamping_info = self.clamping_faces()
        width_tolerance = 15.0  # mm

        # 1. Collect ALL possible pairs from ALL setups into one list
        all_possible_pairs = []
        for setup in clamping_info:
            for pair in setup['face_pairs']:
                pair['parent_setup'] = setup['setup_axis']
                all_possible_pairs.append(pair)

        # 2. Cluster widths -> find a width that satisfies the maximum number of setups
        all_possible_pairs.sort(key=lambda x: x['clamping_width'])

        potential_groups = []
        for p in all_possible_pairs:
            assigned = False
            for group in potential_groups:
                if abs(p['clamping_width'] - np.mean(group['widths'])) <= width_tolerance:
                    group['pairs'].append(p)
                    group['widths'].append(p['clamping_width'])
                    group['unique_setups'].add(p['parent_setup'])
                    assigned = True
                    break
            if not assigned:
                potential_groups.append({
                    'pairs': [p],
                    'widths': [p['clamping_width']],
                    'unique_setups': {p['parent_setup']}
                })

        # Sort by how many unique setups they cover (Descending)-> prioritizes the "Universal Vice"
        potential_groups.sort(key=lambda x: len(x['unique_setups']), reverse=True)

        # 3. ASSIGN SETUPS TO GROUPS
        selected_setups_map = {}  # setup_axis -> best pair within the best group
        remaining_setups = set(s['setup_axis'] for s in clamping_info)
        final_groups_data = []

        for group in potential_groups:
            # If this group contains setups we haven't satisfied yet
            setups_in_group = group['unique_setups'].intersection(remaining_setups)

            if setups_in_group:
                group_setups_indices = []
                group_widths = []
                group_h_maxes = []
                group_h_mins = []
                group_lengths = []

                for s_axis in list(setups_in_group):
                    # Pick the best pair within THIS width group for this setup
                    pairs_for_this_setup = [p for p in group['pairs'] if p['parent_setup'] == s_axis]
                    best_pair_in_group = max(pairs_for_this_setup, key=lambda x: x['stability_score'])

                    # Log data for the final envelope
                    group_setups_indices.append(s_axis)
                    group_widths.append(best_pair_in_group['clamping_width'])
                    group_h_maxes.append(best_pair_in_group['clamping_max_height'])
                    group_h_mins.append(best_pair_in_group['clamping_min_height'])
                    group_lengths.append(best_pair_in_group['clamping_max_length'])

                    remaining_setups.remove(s_axis)

                    f1, f2 = best_pair_in_group['face_axis']
                    print(
                        f"Group Assignment: Setup {s_axis} assigned to width ~{np.mean(group['widths']):.1f} using {f1}/{f2}")

                final_groups_data.append({
                    'setups': group_setups_indices,
                    'widths': group_widths,
                    'h_maxes': group_h_maxes,
                    'h_mins': group_h_mins,
                    'lengths': group_lengths
                })

        # 4. FORMAT OUTPUT
        print("\n" + "=" * 50)
        print("MINIMIZED WORKHOLDING SUGGESTION (VIRTUAL VICES)")
        print("=" * 50)

        final_suggestion = {}
        for i, g in enumerate(final_groups_data):
            name = f"Virtual Vice {i + 1}"
            envelope = {
                'Associated Setups': g['setups'],
                'Required Opening (Width)': round(max(g['widths']), 2),
                'Max Allowable Jaw Height (Tool Safety)': round(min(g['h_maxes']), 2),
                'Min Effective Grip Height (Feature Safety)': round(max(g['h_mins']), 2),
                'Min Jaw Width (Length)': round(max(g['lengths']), 2)
            }
            final_suggestion[name] = envelope

            print(f"\n>>> {name}")
            for k, v in envelope.items():
                print(f"  {k}: {v}")

        return final_suggestion


    # helper visualization
    def visualize_common_area(self, axis1, axis2, common_points):
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        # Helper to plot 3D points
        def add_trace(pts, name, color, opac, size):
            # Check if pts is actually a list/array
            if pts is None or not isinstance(pts, (list, np.ndarray)):
                return
            if len(pts) == 0:
                return

            # Ensure we are only looking at the points, even if a float sneaked into the list
            valid_pts = [p for p in pts if isinstance(p, (list, tuple, np.ndarray))]

            if not valid_pts:
                return

            lengths = set(len(p) for p in valid_pts)
            if len(lengths) > 1:
                print(f"Error in {name}: Inconsistent dimensions {lengths}")
                return

            pts_arr = np.array(valid_pts)
            fig.add_trace(go.Scatter3d(
                x=pts_arr[:, 0], y=pts_arr[:, 1], z=pts_arr[:, 2],
                mode='markers', name=name,
                marker=dict(size=size, color=color, opacity=opac)
            ))

        # 1. Raw grids for each face
        add_trace(self.generate_grid(axis1), f"Grid {axis1}", 'red', 0.1, 2)
        add_trace(self.generate_grid(axis2), f"Grid {axis2}", 'green', 0.1, 2)

        # 2. Common Intersection Points
        add_trace(common_points, "Common Clamping Area", 'blue', 0.8, 4)

        fig.update_layout(
            title=f"Common area for faces {axis1} and {axis2}",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data")
        )
        fig.show()


