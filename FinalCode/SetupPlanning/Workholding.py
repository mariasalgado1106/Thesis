from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import analyze_shape, get_stock_box
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies
from SetupPlanning.Setup_Plan import Setup_Plan

import numpy as np


class Workholding:
    def __init__(self, my_shape):
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)
        (self.xmin, self.ymin, self.zmin, self.xmax, self.ymax, self.zmax,
         self.stock_box_center) = get_stock_box(self.shape)

        self.recognizer = FeatureRecognition(self.shape)
        self.features = self.recognizer.identify_features()
        self.colors_rgb = self.recognizer.colors_rgb

        self.setup_plan = Setup_Plan(self.shape)
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
        clamping_height = 5
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
        dual_axis_map = { #input (setup, face axis) -> output(lenghth and height)
            'z': {'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)},
            '-z': {'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)},
            'x': {'y': (2, 0), '-y': (2, 0), 'z': (1, 0), '-z': (1, 0)},
            '-x': {'y': (2, 0), '-y': (2, 0), 'z': (1, 0), '-z': (1, 0)},
            'y': {'x': (2, 1), '-x': (2, 1), 'z': (0, 1), '-z': (0, 1)},
            '-y': {'x': (2, 1), '-x': (2, 1), 'z': (0, 1), '-z': (0, 1)}
        }
        if not common_pts:
            return 0, 0, 0
        idx_len, idx_height = dual_axis_map[setup][face_axis]
        pts_arr = np.array(common_pts)

        # Define stock_min_h based on the current height index
        is_positive_setup = ['x', 'y', 'z']
        if setup in is_positive_setup:
            stock_min_h = [self.xmax, self.ymax, self.zmax][idx_height]
        else:
            stock_min_h = [self.xmin, self.ymin, self.zmin][idx_height]

        step_size = 0.5

        # 1. Group points by their "length" coordinate (columns)
        columns = {}
        for p in pts_arr:
            l_coord = round(p[idx_len], 2)
            if l_coord not in columns:
                columns[l_coord] = []
            columns[l_coord].append(p[idx_height])

        # 2. For every column, find its "continuous height" from the bottom
        column_heights = []
        for l_coord, heights in columns.items():
            if setup in is_positive_setup:
                h_sorted = sorted(heights, reverse=True)
            else:
                h_sorted = sorted(heights)
            h_limit = h_sorted[0]

            for i in range(1, len(h_sorted)):
                if abs(h_sorted[i] - h_sorted[i - 1]) > (step_size * 1.1):
                    break
                else:
                    h_limit = h_sorted[i]
            column_heights.append(abs(round(stock_min_h - h_limit)))
            continue


        # 3. H_min is the MINIMUM of all those continuous columns
        h_min = min(column_heights) if column_heights else 0

        # max_h is the absolute highest point found anywhere in the grid
        max_h = np.max(pts_arr[:, idx_height]) - stock_min_h
        # max_len is the total horizontal span
        max_len = np.max(pts_arr[:, idx_len]) - np.min(pts_arr[:, idx_len])

        return max_len, h_min, max_h

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
                max_min_pts = {'x': (self.xmin, self.xmax),
                               'y': (self.ymin, self.ymax),
                               'z': (self.zmin, self.zmax)}
                clamping_width = abs(max_min_pts[fa1][1] - max_min_pts[fa1][0])
                idx_v = {'z': 2, '-z': 2, 'x': 0, '-x': 0, 'y': 1, '-y': 1}[setup_axis]
                axis_letter = setup_axis.replace('-', '')
                total_part_height = max_min_pts[axis_letter][1] - max_min_pts[axis_letter][0]
                common_pts, common_area = self.common_parallel_area(fa1, fa2)
                if not common_pts:
                    print(f"  Pairs {fa1}/{fa2}: No common area found.")
                    continue
                max_len, h_min, clamping_top = self.find_height_and_length(common_pts, setup_axis, fa1)
                # GAP AT BOTTOM
                stock_min_h = max_min_pts[axis_letter][0]
                min_h_in_grid = np.min(np.array(common_pts)[:, idx_v]) - stock_min_h

                # --- LOCAL STABILITY SCORE ---
                # Ratio of common area vs stock face area * contact length ratio
                # Helps choose the best face pair for THIS setup
                stock_area = abs(max_min_pts[pf1][1] - max_min_pts[pf1][0]) * \
                             abs(max_min_pts[pf3][1] - max_min_pts[pf3][0])
                # tipping risk-> If h_min is 5mm and total_part_height is 100mm, score is low
                height_stability = h_min / (total_part_height + 1e-6)
                # Penalty if we aren't clamping at the floor (min_h_in_grid > 0)
                floor_penalty = 1.0 if min_h_in_grid < 0.6 else (0.5 / min_h_in_grid)

                stability_score = (common_area / stock_area) * height_stability * floor_penalty* (max_len / (
                            max_min_pts[fa1][1] - max_min_pts[fa1][0]))

                print(f"  Pairs {fa1}/{fa2} -> Width: {clamping_width:.2f}, "
                      f"Area: {common_area:.2f}, Max height: {h_min:.2f}")

                clamping_pairs.append({
                    'face_axis': (fa1, fa2),
                    'common_pts': common_pts,
                    'common_area': common_area,
                    'clamping_width': clamping_width,
                    'clamping_max_length': max_len,
                    'clamping_min_height': h_min,
                    'clamping_max_height': clamping_top,
                    'stability_score': stability_score
                })
            clamping_faces_info.append({
                'setup_axis': setup_axis,
                'face_pairs': clamping_pairs
            })

        return clamping_faces_info

    def final_clamping_suggestion(self):
        clamping_info = self.clamping_faces()
        width_tolerance = 15.0  # mm - Allowable width diff to stay in same vice group

        # 1. If width is common --> boost its score
        all_widths = []
        for setup in clamping_info:
            for pair in setup['face_pairs']:
                all_widths.append(pair['clamping_width'])

        for setup in clamping_info:
            for pair in setup['face_pairs']:
                # Count how many other setups have a similar width
                common_count = sum(1 for w in all_widths if abs(w - pair['clamping_width']) <= width_tolerance)
                pair['global_score'] = pair['stability_score'] + (common_count * 0.1)

        # 2. SELECT BEST PAIR PER SETUP --> choice based on Global Score
        print("\n--- SELECTED BEST PAIR PER SETUP ---")
        selected_setups = []
        for setup in clamping_info:
            if not setup['face_pairs']:
                print(f"Setup {setup['setup_axis']}: SKIPPED (No valid pairs)")
                continue
            best_pair = max(setup['face_pairs'], key=lambda x: x['global_score'])
            print(f"Setup {setup['setup_axis']}: Selected {best_pair['face_axis']} "
                f"(Width: {best_pair['clamping_width']:.2f})")
            selected_setups.append({
                'setup_axis': setup['setup_axis'],
                'data': best_pair
            })

        # 3. GROUP BY width
        selected_setups.sort(key=lambda x: x['data']['clamping_width'])
        vice_groups = []
        for s in selected_setups:
            assigned = False
            for group in vice_groups:
                # Check if width fits the group range
                if abs(s['data']['clamping_width'] - np.mean(group['widths'])) <= width_tolerance:
                    group['setups'].append(s['setup_axis'])
                    group['widths'].append(s['data']['clamping_width'])
                    group['h_mins'].append(s['data']['clamping_min_height'])
                    group['lengths'].append(s['data']['clamping_max_length'])
                    assigned = True
                    break

            if not assigned:
                vice_groups.append({
                    'setups': [s['setup_axis']],
                    'widths': [s['data']['clamping_width']],
                    'h_mins': [s['data']['clamping_min_height']],
                    'lengths': [s['data']['clamping_max_length']]
                })

        # 4. FORMAT OUTPUT
        print("\n" + "=" * 50)
        print("FINAL WORKHOLDING SUGGESTION (VIRTUAL VICES)")
        print("=" * 50)

        final_suggestion = {}
        for i, g in enumerate(vice_groups):
            name = f"Virtual Vice {i + 1}"
            envelope = {
                'Associated Setups': g['setups'],
                'Width Range (Part)': (round(min(g['widths']), 2), round(max(g['widths']), 2)),
                'Required Opening (Min)': round(max(g['widths']), 2),
                'Max Safe Jaw Height': round(min(g['h_mins']), 2),
                'Min Required Jaw Width': round(max(g['lengths']), 2)
            }
            final_suggestion[name] = envelope

            print(f"\n>>> {name}")
            for k, v in envelope.items():
                print(f"  {k}: {v}")

        print("\n" + "=" * 50)
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


