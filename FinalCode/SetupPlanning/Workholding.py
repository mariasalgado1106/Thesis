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

    # ACTUAL Function
    def clamping_faces (self):
        vice_library = {
            'width': 200,
            'height': 50,
            'length': 150
        }

        # Total Area and BBox Area
        total_part_surface_area = sum(self.face_data_list[f]['face_area'] for f in range(len(self.face_data_list)))
        stock_indices = {face['stock_face_idx'] for face in self.stock_faces}
        final_stfaces_area = sum(self.face_data_list[f]['face_area'] for f in range(len(self.face_data_list))
            if f in stock_indices)
        dx = abs(self.xmax - self.xmin)
        dy = abs(self.ymax - self.ymin)
        dz = abs(self.zmax - self.zmin)
        bbox_surface_area = 2 * (dx * dy + dx * dz + dy * dz)
        print(f"Total Area of Stock Faces (Final Geometry): {final_stfaces_area:.2f} mm²")
        print(f"area total: {total_part_surface_area} and bbox area: {bbox_surface_area}")

        perpendicular_axis = {'z': ('x', '-x', 'y', '-y'), '-z': ('x', '-x', 'y', '-y'),
                              'x': ('z', '-z', 'y', '-y'), '-x': ('z', '-z', 'y', '-y'),
                              'y': ('x', '-x', 'z', '-z'), '-y': ('x', '-x', 'z', '-z')}
        clamping_faces_info = []

        # TABLE HEADER
        print("\n" + "=" * 125)
        print(
            f"{'Setup':<8} | {'Pair':<10} | {'Width':<8} | {'Height':<8} | {'Length':<8} | {'H-Ratio':<8} | {'L-Ratio':<8} | {'HangH-L':<8} | {'BArea-R':<8} | {'Status'}")
        print("-" * 125)
        print(
            f"{'LIB':<8} | {'N/A':<10} | {vice_library['width']:<8.2f} | {vice_library['height']:<8.2f} | {vice_library['length']:<8.2f} | {'0.33':<8} | {'0.66':<8} | {'3.00':<8} | {'0.05':<8} | REFERENCE")

        for setup in self.optimized_plan:
            setup_axis = setup['setup']
            pf1,pf2,pf3,pf4 = perpendicular_axis[setup_axis]
            pairs_parallel_faces = {(pf1,pf2), (pf3,pf4)}
            clamping_pairs = []
            print(f"\nSetup {setup_axis}:")
            for fa1,fa2 in pairs_parallel_faces: #fa = face axis
                #print(f"VALIDATING Pair {fa1}/{fa2}.")
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

                # 4. length
                len_axis = {  # input (setup, face axis) -> output(length and height)
                    'z': {'x': 'y', 'y': 'x'},
                    'x': {'y': 'z', 'z': 'y'},
                    'y': {'x': 'z', 'z': 'x'},
                }
                idx_len = len_axis[axis_letter][fa1]
                total_len = abs(max_min_pts[idx_len][1] - max_min_pts[idx_len][0])

                # 5. Area
                ## clamp only up to h_filt
                h_filt = min(h_max, vice_library['height'])
                is_pos = setup_axis in ['x', 'y', 'z']
                ref_floor = max_min_pts[setup_axis.replace('-', '')][1 if is_pos else 0]
                ## grids for both faces
                grid1 = self.generate_grid(fa1)
                grid2 = self.generate_grid(fa2)
                ## Filter points on face 1 and face 2 based on h_filt
                pts_f1 = [p for p in grid1 if abs(p[idx_height] - ref_floor) <= h_filt]
                pts_f2 = [p for p in grid2 if abs(p[idx_height] - ref_floor) <= h_filt]
                total_clamped_area = (len(pts_f1) + len(pts_f2)) * (0.5 ** 2)

                # 6. RATIOS
                h_ratio = h_max / total_part_height
                len_ratio = max_len/total_len
                hanging_height = total_part_height - h_filt
                hanging_height_length_ratio = (hanging_height)/total_len
                bbox_area_ratio = total_clamped_area / bbox_surface_area

                # 7. Flagging Logic
                is_valid = (h_ratio >= 0.33 and len_ratio >= 0.66 and bbox_area_ratio >= 0.05
                            and hanging_height_length_ratio <= 3)
                status = "PASS" if is_valid else "WARN"
                # Print Row
                pair_str = f"{fa1}/{fa2}"
                print(
                    f"{setup_axis:<8} | {pair_str:<10} | {clamping_width:<8.2f} | {h_max:<8.2f} | {max_len:<8.2f} | {h_ratio:<8.2f} | {len_ratio:<8.2f} | {hanging_height_length_ratio:<8.2f} | {bbox_area_ratio:<8.2f} | {status}")

                clamping_pairs.append({
                    'face_axis': (fa1, fa2),
                    'clamping_width': clamping_width,
                    'h_ratio': h_ratio,
                    'len_ratio': len_ratio,
                    'bbox_area_ratio': bbox_area_ratio,
                    'hanging_height_length_ratio': hanging_height_length_ratio,
                    'status': status,
                    'stability_score': total_clamped_area * h_ratio
                })
            clamping_faces_info.append({
                'setup_axis': setup_axis,
                'face_pairs': clamping_pairs
            })
        print("=" * 125 + "\n")
        return clamping_faces_info

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
