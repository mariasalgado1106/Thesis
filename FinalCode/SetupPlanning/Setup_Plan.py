from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import analyze_shape, get_stock_box
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies

import numpy as np
import itertools

from OCC.Core.BRepProj import BRepProj_Projection
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Lin
from OCC.Core.BRepClass3d import BRepClass3d_SolidClassifier
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

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

    #### Grouping ####
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

    ### Define stock faces and their characteristics ####

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

    #### define the points for locators based on grid ###

    def generate_locating_grid(self, PLFs, axis, step_size=1.0):
        axis_map = {
            'z': (0, 1, 2), '-z': (0, 1, 2),
            'x': (1, 2, 0), '-x': (1, 2, 0),
            'y': (0, 2, 1), '-y': (0, 2, 1)
        }
        idx1, idx2, fixed_idx = axis_map[axis.lower()]

        # bounding box
        xmin, ymin, zmin, xmax, ymax, zmax, _ = get_stock_box(self.shape)
        bounds = [(xmin, xmax), (ymin, ymax), (zmin, zmax)]
        dim1_range = np.arange(bounds[idx1][0], bounds[idx1][1], step_size)
        dim2_range = np.arange(bounds[idx2][0], bounds[idx2][1], step_size)

        grid_points = []

        # check each point against PLF meshes (if it's solid or empty)
        for v1 in dim1_range:
            for v2 in dim2_range:
                is_on_material = False
                for plf in PLFs:
                    f_idx = plf['PLF_idx']
                    vertices = self.face_data_list[f_idx]['mesh_vertices']
                    triangles = self.face_data_list[f_idx]['mesh_triangles']
                    # Point-in-Triangle check (projected to 2D)
                    if self._is_point_in_face_mesh(v1, v2, vertices, triangles, idx1, idx2):
                        is_on_material = True
                        break

                if is_on_material:
                    # Store the actual 3D coordinate of the valid point
                    h = PLFs[0]['PLF_center'][fixed_idx] #use 3rd coord from center of faces
                    pnt = [0, 0, 0]
                    pnt[idx1], pnt[idx2], pnt[fixed_idx] = v1, v2, h
                    grid_points.append(pnt)
        return grid_points

    def _is_point_in_face_mesh(self, u, v, vertices, triangles, idx1, idx2):
        """Helper to check if 2D point (u,v) is inside any triangle of the mesh."""
        for tri in triangles:
            # 2D projection of the triangle vertices
            p1 = vertices[tri[0]]
            p2 = vertices[tri[1]]
            p3 = vertices[tri[2]]

            if self._point_in_triangle_2d(u, v, p1[idx1], p1[idx2],
                                          p2[idx1], p2[idx2],
                                          p3[idx1], p3[idx2]):
                return True
        return False

    def _point_in_triangle_2d(self, u, v, x1, y1, x2, y2, x3, y3):
        denominator = ((y2 - y3) * (x1 - x3) + (x3 - x2) * (y1 - y3))
        # Avoid division by zero for degenerate triangles
        if abs(denominator) < 1e-9:
            return False
        # Calculate Barycentric weights w1, w2, w3
        w1 = ((y2 - y3) * (u - x3) + (x3 - x2) * (v - y3)) / denominator
        w2 = ((y3 - y1) * (u - x3) + (x1 - x3) * (v - y3)) / denominator
        w3 = 1.0 - w1 - w2
        # Point is inside if all weights are between 0 and 1
        return (w1 >= 0) and (w2 >= 0) and (w3 >= 0)

    def get_part_cog(self): #center of gravity of final part
        props = GProp_GProps()
        brepgprop.VolumeProperties(self.shape, props)
        cog_occ = props.CentreOfMass()
        return np.array([cog_occ.X(), cog_occ.Y(), cog_occ.Z()])

    def calculate_2d_area(self, p1, p2, p3, dims):
        x1, y1 = p1[dims[0]], p1[dims[1]]
        x2, y2 = p2[dims[0]], p2[dims[1]]
        x3, y3 = p3[dims[0]], p3[dims[1]]
        return abs((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)) / 2.0)

    def find_PLF_locators(self, grid_points, axis, step_size=1.0):
        cog = self.get_part_cog()
        axis_map = {'z': (0, 1), '-z': (0, 1), 'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)}
        idx1, idx2 = axis_map[axis.lower()]

        pts_array = np.array(grid_points)
        dim1_min, dim1_max = np.min(pts_array[:, idx1]), np.max(pts_array[:, idx1])
        dim2_min, dim2_max = np.min(pts_array[:, idx2]), np.max(pts_array[:, idx2])

        # 1. offset to border of grid (internal as well)
        offset_out = 8
        offset_in = 4

        grid_set = set((round(p[idx1], 3), round(p[idx2], 3)) for p in grid_points)
        safe_points = []
        for p in grid_points:
            if not (dim1_min + offset_out - 1e-3 <= p[idx1] <= dim1_max - offset_out + 1e-3 and
                    dim2_min + offset_out - 1e-3 <= p[idx2] <= dim2_max - offset_out + 1e-3):
                continue
            test_points = [
                (round(p[idx1] + offset_in, 3), round(p[idx2], 3)),
                (round(p[idx1] - offset_in, 3), round(p[idx2], 3)),
                (round(p[idx1], 3), round(p[idx2] + offset_in, 3)),
                (round(p[idx1], 3), round(p[idx2] - offset_in, 3))
            ]
            if all(tp in grid_set for tp in test_points):
                safe_points.append(p)

        print (f"Safe points: {len(safe_points)}")
        #self.visualize_setup_results(safe_points)
        if len(safe_points) < 3:
            print("Fallback: Offset too large.")
            safe_points = grid_points

        # 2. ITERATIVE SEARCH FOR BEST BALANCED TRIANGLE
        # Sort safe points by distance from CoG to get "corner" candidates
        corner_candidates = sorted(safe_points,
            key=lambda p: np.sqrt((p[idx1] - cog[idx1]) ** 2 + (p[idx2] - cog[idx2]) ** 2),
            reverse=True)
        best_trio = None
        max_area = -1
        is_balanced = False
        # We check combinations of the best corner candidates (Top 15)
        # 15 points = 455 combinations. Very fast to check.
        import itertools
        for trio in itertools.combinations(corner_candidates[:15], 3):
            p1, p2, p3 = trio
            # Check Balance
            balanced = self._point_in_triangle_2d(cog[idx1], cog[idx2],
                                                  p1[idx1], p1[idx2],
                                                  p2[idx1], p2[idx2],
                                                  p3[idx1], p3[idx2])
            # Check Area
            area = self.calculate_2d_area(p1, p2, p3, (idx1, idx2))

            # We want the biggest area that is balanced
            if balanced and area > max_area:
                max_area = area
                best_trio = (p1, p2, p3)
                is_balanced = True

        # 3. Final Fallback
        # If NO balanced triangle was found in the corners, take the absolute biggest triangle
        if not best_trio:
            p1 = corner_candidates[0]
            p2 = max(safe_points, key=lambda p: np.linalg.norm(p[[idx1, idx2]] - p1[[idx1, idx2]]))
            p3 = max(safe_points, key=lambda p: self.calculate_2d_area(p1, p2, p, (idx1, idx2)))
            best_trio = (p1, p2, p3)
            is_balanced = False  # Still unbalanced, but at least we have a solution

        return best_trio, is_balanced

    def validate_workholding (self, axis):
        # apply 321 technique based on the final part (worst case scenario)
        PLFs = []
        PLF_locators = None
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
        # 2.1. Validate based on area ratio (30% of original)
        original_area = self.get_original_stock_area(axis)
        if original_area > 0:
            area_ratio = (PLF_total_area / original_area) * 100
            print(f"Total PLF Area: {PLF_total_area:.2f} / Original: {original_area:.2f} ({area_ratio:.1f}%)")
            if area_ratio < 30:
                print(f"WARNING: Stability compromised! Only {area_ratio:.1f}% of the base remains.")
                validated = False

        # 2.2 Identify 3 specific locator points based on centers
        nr_PLFs = len(PLFs)
        grid_pts = self.generate_locating_grid(PLFs, axis)
        if len(grid_pts) >= 3:
            locators, balanced = self.find_PLF_locators(grid_pts, axis)
            PLF_locators = locators  # Assign the found points to PLF for the return statement
            print(f"Primary Locators: {locators}")
            print(f"CoG Balanced: {balanced}")


        return PLF_locators, SLF, TLF





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

    def setup_order(self, features_of_setup, current_setup_axis):
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
        pending = list(features_of_setup)  # feat not done still
        previous_was_hole = 0

        while pending:
            available = []  # available to be considered and produced now
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
            grouped = self.group_by_feat_type(available)  # groups only the available ones
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

    def visualize_setup_results(self, grid_points, locators=None, cog=None):
        import plotly.graph_objects as go
        import numpy as np

        data = []

        # 1. Plot the Grid Points
        if grid_points:
            pts = np.array(grid_points)
            grid_trace = go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode='markers',
                name='Valid Grid',
                marker=dict(size=2, color='red', opacity=0.3)
            )
            data.append(grid_trace)

        # 2. Plot the Locators (Change symbol to 'circle')
        if locators:
            locs = np.array(locators)
            loc_trace = go.Scatter3d(
                x=locs[:, 0], y=locs[:, 1], z=locs[:, 2],
                mode='markers+text',
                name='3-2-1 Locators',
                text=["P1", "P2", "P3"],
                textposition="top center",
                marker=dict(size=8, color='blue', symbol='circle')  # Fix here
            )
            data.append(loc_trace)

        # 3. Plot the CoG
        if cog is not None:
            cog_trace = go.Scatter3d(
                x=[cog[0]], y=[cog[1]], z=[cog[2]],
                mode='markers',
                name='Part CoG',
                marker=dict(size=10, color='green', symbol='diamond')  # Fix here
            )
            data.append(cog_trace)

        fig = go.Figure(data=data)
        fig.update_layout(
            title="Workholding Validation Results",
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Z',
                aspectmode='data'
            )
        )
        fig.show()
