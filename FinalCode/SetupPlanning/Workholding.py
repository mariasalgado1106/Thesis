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

    def VicesLibrary (self):
        vices = [
            {'name': 'Standard 6-inch', 'jaw_width': 152.4, 'max_opening': 150.0, 'jaw_height': 44.0},
            {'name': 'Compact 4-inch', 'jaw_width': 101.6, 'max_opening': 100.0, 'jaw_height': 35.0}
        ]
        return vices

    # Find 2 pairs of parallel faces and their common points in a plane? ->
    # ->use this common area to help determine which ones are best
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
        clamping_height = 20
        dim1_range = np.arange(bounds[idx1][0], bounds[idx1][1], step_size)
        dim2_range = np.arange(bounds[idx2][0], bounds[idx2][0] + clamping_height , step_size)
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


    def clamping_faces (self):
        perpendicular_axis = {'z': ('x', '-x', 'y', '-y'), '-z': ('x', '-x', 'y', '-y'),
                              'x': ('z', '-z', 'y', '-y'), '-x': ('z', '-z', 'y', '-y'),
                              'y': ('x', '-x', 'z', '-z'), '-y': ('x', '-x', 'z', '-z')}
        for setup in self.optimized_plan:
            setup_axis = setup['setup']
            pf1,pf2,pf3,pf4 = perpendicular_axis[setup_axis]
            pairs_parallel_faces = {(pf1,pf2), (pf3,pf4)}
            for fa1,fa2 in pairs_parallel_faces: #fa = face axis
                max_min_pts = {'x': (self.xmin, self.xmax),
                               'y': (self.ymin, self.ymax),
                               'z': (self.zmin, self.zmax)}
                min, max = max_min_pts[fa1]
                clamping_width = max - min
                common_pts, common_area =self.common_parallel_area(fa1, fa2)


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


