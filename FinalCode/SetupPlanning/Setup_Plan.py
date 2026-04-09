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
from networkx.generators.harary_graph import hkn_harary_graph


class Setup_Plan:
    def __init__(self, my_shape, recognizer=None):
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)

        self.recognizer = recognizer if recognizer else FeatureRecognition(self.shape)
        self.features = self.recognizer.identify_features()
        self.colors_rgb = self.recognizer.colors_rgb

        self.tad_extractor = TAD_Extraction(self.shape, recognizer=self.recognizer)
        self.dep_extractor = Dependencies(self.shape, recognizer=self.recognizer)
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

    def generate_locating_grid(self, xLFs, axis, step_size=1.0):
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
                for xlf in xLFs:
                    f_idx = xlf['Face_idx']
                    vertices = self.face_data_list[f_idx]['mesh_vertices']
                    triangles = self.face_data_list[f_idx]['mesh_triangles']
                    # Point-in-Triangle check (projected to 2D)
                    if self._is_point_in_face_mesh(v1, v2, vertices, triangles, idx1, idx2):
                        is_on_material = True
                        break

                if is_on_material:
                    # Store the actual 3D coordinate of the valid point
                    h = xLFs[0]['Face_center'][fixed_idx] #use 3rd coord from center of faces
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

    def define_safe_pts_grid (self, axis, grid_points, offset_out, offset_in):
        axis_map = {'z': (0, 1), '-z': (0, 1), 'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)}
        idx1, idx2 = axis_map[axis.lower()]
        pts_array = np.array(grid_points)
        dim1_min, dim1_max = np.min(pts_array[:, idx1]), np.max(pts_array[:, idx1])
        dim2_min, dim2_max = np.min(pts_array[:, idx2]), np.max(pts_array[:, idx2])

        grid_set = set((round(p[idx1], 3), round(p[idx2], 3)) for p in grid_points)
        safe_points = []
        for p in grid_points:
            if not (dim1_min + offset_out - 1e-3 <= p[idx1] <= dim1_max - offset_out + 1e-3 and
                    dim2_min + offset_out - 1e-3 <= p[idx2] <= dim2_max - offset_out + 1e-3):
                continue
            is_point_safe = True
            i=offset_in
            while i > 0:
                j = round(i/2)
                test_points = [
                    (round(p[idx1] + i, 3), round(p[idx2], 3)),
                    (round(p[idx1] - i, 3), round(p[idx2], 3)),
                    (round(p[idx1], 3), round(p[idx2] + i, 3)),
                    (round(p[idx1], 3), round(p[idx2] - i, 3)),
                    (round(p[idx1] + j, 3), round(p[idx2] + j, 3)),
                    (round(p[idx1] - j, 3), round(p[idx2] + j, 3)),
                    (round(p[idx1] + j, 3), round(p[idx2] - j, 3)),
                    (round(p[idx1] - j, 3), round(p[idx2] - j, 3))
                ]
                # If any of these 8 points are NOT in the grid, it's near an edge/hole
                if not all(tc in grid_set for tc in test_points):
                    is_point_safe = False
                    break
                i -= 1

            if is_point_safe:
                safe_points.append(p)

        return idx1, idx2, safe_points

    def find_locators(self, grid_points_PLF, PLF_axis, grid_points_SLF, SLF_axis, grid_points_TLF, TLF_axis):
        cog = self.get_part_cog()
        axis_map = {'z': (0, 1), '-z': (0, 1), 'x': (1, 2), '-x': (1, 2), 'y': (0, 2), '-y': (0, 2)}

        ######## 1. PLF LOCATORS ###########
        tries = 0
        offset_plf = 15
        while tries <= 3:
            # 1.1. Safe points grid
            idx1, idx2, safe_points = self.define_safe_pts_grid(PLF_axis, grid_points_PLF, offset_plf, offset_plf)
            if len(safe_points) < 3:
                tries += 1
                offset_plf -= 5
                continue

            # 1.2. QUADRANT SAMPLING FOR BETTER BALANCE
            # Separate points into 4 lists based on their position relative to CoG
            q_lists = [
                [p for p in safe_points if p[idx1] >= cog[idx1] and p[idx2] >= cog[idx2]],  # Q1
                [p for p in safe_points if p[idx1] < cog[idx1] and p[idx2] >= cog[idx2]],  # Q2
                [p for p in safe_points if p[idx1] < cog[idx1] and p[idx2] < cog[idx2]],  # Q3
                [p for p in safe_points if p[idx1] >= cog[idx1] and p[idx2] < cog[idx2]]  # Q4
            ]
            sorted_quadrants = []
            for q in q_lists:
                sorted_q = sorted(q, key=lambda p: np.sqrt((p[idx1] - cog[idx1]) ** 2 + (p[idx2] - cog[idx2]) ** 2),
                                  reverse=True)
                sorted_quadrants.append(sorted_q)
            k = 3  # Start with top 3 furthest points per quadrant
            max_k = 20  # Safety limit
            PLF_locators = None
            max_area = -1
            is_balanced = False
            while k <= max_k and not is_balanced:
                sampling_pool = []
                for sq in sorted_quadrants:
                    sampling_pool.extend(sq[:k])
                unique_pool = []  # avoid duplicates from previous iteration
                [unique_pool.append(p) for p in sampling_pool if p not in unique_pool]
                # Check all combinations in the current pool
                for trio in itertools.combinations(unique_pool, 3):
                    p1, p2, p3 = trio
                    balanced = self._point_in_triangle_2d(cog[idx1], cog[idx2],
                                                          p1[idx1], p1[idx2],
                                                          p2[idx1], p2[idx2],
                                                          p3[idx1], p3[idx2])

                    if balanced:
                        area = self.calculate_2d_area(p1, p2, p3, (idx1, idx2))
                        if area > max_area:
                            max_area = area
                            PLF_locators = (p1, p2, p3)
                            is_balanced = True

                if is_balanced:
                    print(f"Success: Balanced solution found at k={k}")
                    break

                k += 2  # Increase the depth of the search
            if is_balanced:
                print(f"Success: Balanced solution found at try={tries}")
                break
            tries += 1
            offset_plf -= 5


        # 1.3. FINAL FALLBACK (If still no balance, just take the biggest possible)
        if not PLF_locators:
            print("Final Fallback: No balanced solution possible. Choosing max area.")
            p1 = sorted_quadrants[0][0] if sorted_quadrants[0] else safe_points[0]
            p2 = max(safe_points, key=lambda p: np.sqrt((p[idx1] - p1[idx1])**2 + (p[idx2] - p1[idx2])**2))
            p3 = max(safe_points, key=lambda p: self.calculate_2d_area(p1, p2, p, (idx1, idx2)))
            PLF_locators = (p1, p2, p3)
            is_balanced = False  # Still unbalanced, but at least we have a solution


        ######## 2. SLF LOCATORS ###########
        offset_slf = 15
        idx1_slf, idx2_slf, safe_points_slf = self.define_safe_pts_grid(SLF_axis, grid_points_SLF,
                                                                        offset_slf, offset_slf-5)
        while len(safe_points_slf) < 2:
            offset_slf -= 5
            if offset_slf < 0:
                offset_slf = 0
            idx1_slf, idx2_slf, safe_points_slf = self.define_safe_pts_grid (SLF_axis, grid_points_SLF,
                                                                         offset_slf, offset_slf)
        # 2.1 Determine height, normal and width axis
        plf_normal_idx = list({0, 1, 2} - set(axis_map[PLF_axis.lower()]))[0] #height
        slf_normal_idx = list({0, 1, 2} - set(axis_map[SLF_axis.lower()]))[0]
        search_idx = list({0, 1, 2} - {plf_normal_idx, slf_normal_idx})[0] #width, tlf axis

        # 2.2. Filter to Parallel Points (same height)
        unique_heights = sorted(list(set(p[plf_normal_idx] for p in safe_points_slf)))
        median_height = unique_heights[len(unique_heights) // 2]
        parallel_points = [p for p in safe_points_slf if p[plf_normal_idx] == median_height]
        if len(parallel_points) < 2:
            parallel_points = safe_points_slf

        # 3. Maximize Distance along the width (search_idx)
        p1_slf = min(parallel_points, key=lambda p: p[search_idx])
        p2_slf = max(parallel_points, key=lambda p: p[search_idx])
        SLF_locators = (p1_slf, p2_slf)

        ######## 3. TLF LOCATOR  ###########
        offset_tlf = 10
        idx1_tlf, idx2_tlf, safe_points_tlf = self.define_safe_pts_grid(TLF_axis, grid_points_TLF,
                                                                        offset_tlf, offset_tlf-3)
        while len(safe_points_tlf) < 1:
            offset_tlf -= 5
            idx1_tlf, idx2_tlf, safe_points_tlf = self.define_safe_pts_grid(TLF_axis, grid_points_TLF,
                                                                            offset_tlf, offset_tlf)
        # Pick the point closest to the geometric center of the safe area for max stability
        avg_dim1 = np.mean([p[idx1_tlf] for p in safe_points_tlf])
        avg_dim2 = np.mean([p[idx2_tlf] for p in safe_points_tlf])

        p1_tlf = min(safe_points_tlf, key=lambda p: (p[idx1_tlf] - avg_dim1) ** 2 +
                                                    (p[idx2_tlf] - avg_dim2) ** 2)
        TLF_locators = (p1_tlf,)

        #self.visualize_safe_points(grid_points_PLF, safe_points, PLF_axis)
        #self.visualize_safe_points(grid_points_SLF, safe_points_slf, SLF_axis)
        #self.visualize_safe_points(grid_points_SLF, parallel_points, SLF_axis)
        #self.visualize_safe_points(grid_points_TLF, safe_points_tlf, TLF_axis)

        return PLF_locators, is_balanced, SLF_locators, TLF_locators

    def validate_workholding (self, axis):
        # apply 321 technique based on the final part (worst case scenario)
        PLFs, SLFs, TLFs = [], [], []
        PLF, SLF, TLF = [], [], []

        ##########################################################
        # 1. Get stock faces and possible base face (Primary Locating Face) & SLF & TLF
        stock_faces = self.define_stock_faces_list()
        PLF_total_area = 0
        # 1.1. Perpendicular faces based on axis (to determine slf and tlf)
        perpendicular_axis = {'z': ('x', '-x', 'y', '-y'), '-z': ('x', '-x', 'y', '-y'),
                              'x': ('z', '-z', 'y', '-y'), '-x': ('z', '-z', 'y', '-y'),
                              'y': ('x', '-x', 'z', '-z'), '-y': ('x', '-x', 'z', '-z')}
        perp_data = {pa: {'faces': [], 'total_area': 0} for pa in perpendicular_axis[axis]}
        for stock_face in stock_faces:
            sf_axis = stock_face['opposite_TAD']
            sf_idx = stock_face['stock_face_idx']
            sf_center = self.face_data_list[sf_idx]['face_center']
            sf_area = stock_face['area']
            # 1.2. Determine PLFs
            if sf_axis == axis:
                PLFs.append({ #list of the coplanar faces in that plane
                    'Face_idx': sf_idx,
                    'Face_center': sf_center,
                    'PLF_area': sf_area
                    })
                PLF_total_area = PLF_total_area + sf_area
                #print(f"Face {stock_face['stock_face_idx']} is a candidate for PLF of Setup of TAD {axis}")
            # 1.3. Determine possible SLFs
            elif sf_axis in perp_data:
                perp_data[sf_axis]['faces'].append({
                    'Face_idx': sf_idx,
                    'Face_center': sf_center,
                    'SLF_area': sf_area
                })
                perp_data[sf_axis]['total_area'] += sf_area


        ###############################################################
        # 2. Get SLFs and TLFs faces (perpendicularity & biggest area)
        slf_axis = max(perp_data, key=lambda k: perp_data[k]['total_area'])
        SLFs = perp_data[slf_axis]['faces']

        pa1, pa2, pa3, pa4 = perpendicular_axis[axis]
        new_perp_axis = {pa1: (pa3, pa4), pa2: (pa3, pa4), pa3: (pa1, pa2), pa4: (pa1, pa2)}
        perp_data2 = {pa: {'faces': [], 'total_area': 0} for pa in new_perp_axis[slf_axis]}
        for stock_face in stock_faces:
            sf_axis = stock_face['opposite_TAD']
            sf_idx = stock_face['stock_face_idx']
            sf_center = self.face_data_list[sf_idx]['face_center']
            sf_area = stock_face['area']
            if sf_axis in perp_data2:
                perp_data2[sf_axis]['faces'].append({
                    'Face_idx': sf_idx,
                    'Face_center': sf_center,
                    'TLF_area': sf_area
                })
                perp_data2[sf_axis]['total_area'] += sf_area
        tlf_axis = max(perp_data2, key=lambda k: perp_data2[k]['total_area'])
        TLFs = perp_data2[tlf_axis]['faces']

        ###############################################################
        # 3. Validate areas based on area ratio (30% of original)
        validated = True
        # 3.1. PLF
        original_area = self.get_original_stock_area(axis)
        if original_area > 0:
            area_ratio = (PLF_total_area / original_area) * 100
            if area_ratio < 30:
                print(f"WARNING: Stability compromised! Only {area_ratio:.1f}% of the base remains.")
                validated = False

        # 3.2. SLF
        original_area = self.get_original_stock_area(slf_axis)
        SLF_total_area = perp_data[slf_axis]['total_area']
        if original_area > 0:
            area_ratio = (SLF_total_area / original_area) * 100
            if area_ratio < 30:
                print(f"WARNING: Stability compromised! Only {area_ratio:.1f}% of the SLF remains.")
                validated = False

        # 3.3. TLF
        original_area = self.get_original_stock_area(tlf_axis)
        TLF_total_area = perp_data[tlf_axis]['total_area']
        if original_area > 0:
            area_ratio = (TLF_total_area / original_area) * 100
            if area_ratio < 30:
                print(f"WARNING: Stability compromised! Only {area_ratio:.1f}% of the TLF remains.")
                validated = False

        #############################################################
        # 4. Find locator points
        PLF_grid_pts = self.generate_locating_grid(PLFs, axis)
        SLF_grid_pts = self.generate_locating_grid(SLFs, slf_axis)
        TLF_grid_pts = self.generate_locating_grid(TLFs, tlf_axis)
        PLF_locators, balanced, SLF_locators, TLF_locators = self.find_locators(PLF_grid_pts, axis,
                                                                                SLF_grid_pts, slf_axis,
                                                                                TLF_grid_pts, tlf_axis)
        # 4.1. Find 3 locators in PLF
        if len(PLF_grid_pts) >= 3 and validated:
            if not balanced:
                validated = False
                print("!!Couldn't find a balanced solution in PLF!!")
            else:
                PLF = ({
                    'PLF_faces': PLFs,
                    'PLF_locators': PLF_locators
                })
                #print(f"Primary Locators: {PLF_locators} & CoG Balanced?: {balanced}")

        # 4.2. Find 2 locators in SLF
        if len(SLF_grid_pts) >= 2 and validated:
            SLF = ({
                'SLF_faces': SLFs,
                'SLF_locators': SLF_locators
            })
            #print(f"Secondary Locators: {SLF_locators}")

        # 4.3. Find 1 locator in TLF
        if len(TLF_grid_pts) >= 1 and validated:
            TLF = ({
                'TLF_faces': TLFs,
                'TLF_locators': TLF_locators
            })
            #print(f"Terciary Locator: {TLF_locators}")


        return PLF, SLF, TLF, validated


    def generate_optimized_plan(self):
        ### 1. features grouped by tads
        groups = self.group_by_tads()

        ### 2. order tads (max features 1st)
        # sort by the length of the feature list in each group
        sorted_setups = sorted([axis for axis in groups if axis != "INACCESSIBLE"],
            key=lambda x: len(groups[x]), reverse=True)

        optimized_plan = [] # setups in order, respective features, PLF, SLF, TLF
        already_planned = set() #features that were already done, for filtering after

        # Initialize tracking for sharp edges
        extra_setup_tracker = []
        for feat in self.features:
            if feat['feature_type'] in ['feat_slot_blind', 'feat_step_blind']:
                extra_setup_tracker.append({
                    'feat_idx': feat['feat_idx'],
                    'remaining_setups': 2,
                    'tads': [t['axis'] for t in
                             next(f['tads'] for f in self.feature_info if f['feat_idx'] == feat['feat_idx'])]
                })

        print("\n" + "=" * 50)
        print("GENERATING OPTIMIZED PROCESS PLAN")
        print("=" * 50)

        for axis in sorted_setups:
            # 1. Primary features for this setup that haven't been assigned to a previous setup
            features_to_order = [f for f in groups[axis] if f['feat_idx'] not in already_planned]
            # 2. Check if we need to keep going just for sharp edges
            active_sharp_reqs = [s for s in extra_setup_tracker if s['remaining_setups'] > 0 and axis in s['tads']]

            if not features_to_order and not active_sharp_reqs:
                continue

            is_extra_setup_only = len(features_to_order) == 0
            if is_extra_setup_only:
                print(f"\n>>> Creating EXTRA Setup: {axis} (Sharp Edges Requirement)")
            else:
                print(f"\n>>> Planning Setup: {axis} ({len(features_to_order)} features)")


            print(f"\n>>> Planning Setup: {axis} ({len(features_to_order)} features)")

            ### 3. Validate setup (check the workholding and faces used)
            PLF, SLF, TLF, validated = self.validate_workholding(axis)
            if not validated:
                continue
            print (f"!!!!!!!!!!SETUP {axis} VALIDATED!!!!!!!!!!")

            ### 4. Sharp Edges
            # We find features that can be machined in this axis and still need setups
            extra_features_ids = []
            for item in extra_setup_tracker:
                if item['remaining_setups'] > 0 and axis in item['tads']:
                    # Only add to extra column if it's not already a primary feature in THIS setup
                    if not any(f['feat_idx'] == item['feat_idx'] for f in features_to_order):
                        extra_features_ids.append(item['feat_idx'])
                    # Decrement counter because we are using this axis
                    item['remaining_setups'] -= 1

            # 5. sequence the features within this TAD (not relevant but it's done)
            ordered_sequence = self.order_in_setup(features_to_order, axis)

            # Store the result
            optimized_plan.append({
                'setup': axis,
                'sequence': [f['feat_idx'] for f in ordered_sequence],
                'extra_features': extra_features_ids,
                'is_extra_only': is_extra_setup_only,
                'PLF': PLF,
                'SLF': SLF,
                'TLF': TLF
            })

            # Mark these as done so they aren't produced twice in another setup
            for f in ordered_sequence:
                already_planned.add(f['feat_idx'])

        print("\n" + "=" * 80)
        print(f"{'SETUP':<10} | {'PRIMARY SEQUENCE':<30} | {'SHARP EDGE EXTRAS'}")
        print("-" * 80)
        for step in optimized_plan:
            setup_label = step['setup']
            if step['is_extra_only']:
                setup_label += " (EXTRA)"

            primary_seq = str(step['sequence'])
            extra_seq = str(step['extra_features']) if step['extra_features'] else "-"

            print(f"{setup_label:<10} | {primary_seq:<30} | {extra_seq}")

        print("=" * 80 + "\n")

        return optimized_plan



    def order_in_setup(self, features_of_setup, current_setup_axis):
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
                #print(f"area inicial = {f['priority_area']} vs nova {new}")
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

    def visualize_safe_points(self, raw_grid, safe_points, axis_label):
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        # 1. Plot Raw Grid (Light Red)
        raw_pts = np.array(raw_grid)
        fig.add_trace(go.Scatter3d(
            x=raw_pts[:, 0], y=raw_pts[:, 1], z=raw_pts[:, 2],
            mode='markers',
            name='Raw Grid Points',
            marker=dict(size=2, color='red', opacity=0.2)
        ))

        # 2. Plot Safe Points (Solid Blue)
        if safe_points:
            s_pts = np.array(safe_points)
            fig.add_trace(go.Scatter3d(
                x=s_pts[:, 0], y=s_pts[:, 1], z=s_pts[:, 2],
                mode='markers',
                name='Safe Points (After Offset)',
                marker=dict(size=4, color='blue', opacity=0.8)
            ))

        fig.update_layout(
            title=f"Safe Point Analysis: {axis_label} Axis",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data")
        )
        fig.show()

    def visualize_setup_3d(self, PLF_locs=None, SLF_locs=None, TLF_locs=None, cog=None):
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        # 1. Show the Part Mesh (Grey/Translucent for context)
        all_vertices = []
        all_triangles = []
        vertex_offset = 0

        for face_data in self.face_data_list:
            vertices = face_data.get('mesh_vertices', [])
            triangles = face_data.get('mesh_triangles', [])
            if not vertices: continue

            all_vertices.extend(vertices)
            for tri in triangles:
                all_triangles.append([tri[0] + vertex_offset, tri[1] + vertex_offset, tri[2] + vertex_offset])
            vertex_offset += len(vertices)

        if all_vertices:
            all_vertices = np.array(all_vertices)
            all_triangles = np.array(all_triangles)
            fig.add_trace(go.Mesh3d(
                x=all_vertices[:, 0], y=all_vertices[:, 1], z=all_vertices[:, 2],
                i=all_triangles[:, 0], j=all_triangles[:, 1], k=all_triangles[:, 2],
                color='rgb(200, 200, 200)', opacity=0.3, name='Part Body', showlegend=True
            ))

        # 2. Helper to add locator groups as spheres
        def add_locators(locs, name, color, labels):
            if locs:
                l_pts = np.array(locs)
                fig.add_trace(go.Scatter3d(
                    x=l_pts[:, 0], y=l_pts[:, 1], z=l_pts[:, 2],
                    mode='markers+text', name=name,
                    text=labels, textposition="top center",
                    marker=dict(size=10, color=color, symbol='circle',
                                line=dict(width=2, color='white'))
                ))

        # Plot the 3-2-1 Points
        add_locators(PLF_locs, "Primary (PLF)", "blue", ["P1", "P2", "P3"])
        add_locators(SLF_locs, "Secondary (SLF)", "red", ["S1", "S2"])
        add_locators(TLF_locs, "Tertiary (TLF)", "green", ["T1"])

        # 3. Plot the CoG (Optional, but helpful for balance context)
        if cog is not None:
            fig.add_trace(go.Scatter3d(
                x=[cog[0]], y=[cog[1]], z=[cog[2]],
                mode='markers', name='Part CoG',
                marker=dict(size=12, color='purple', symbol='diamond')
            ))

        fig.update_layout(
            title="3-2-1 Workholding Configuration",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            width=1000, height=800
        )
        fig.show()

    def visualize_all_setups_3d(self, optimized_plan):
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        # 1. Part Body (Opacity 0.5)
        all_vertices = []
        all_triangles = []
        vertex_offset = 0
        for face_data in self.face_data_list:
            vertices = face_data.get('mesh_vertices', [])
            triangles = face_data.get('mesh_triangles', [])
            if not vertices: continue
            all_vertices.extend(vertices)
            for tri in triangles:
                all_triangles.append([tri[0] + vertex_offset, tri[1] + vertex_offset, tri[2] + vertex_offset])
            vertex_offset += len(vertices)

        if all_vertices:
            all_vertices = np.array(all_vertices)
            all_triangles = np.array(all_triangles)
            fig.add_trace(go.Mesh3d(
                x=all_vertices[:, 0], y=all_vertices[:, 1], z=all_vertices[:, 2],
                i=all_triangles[:, 0], j=all_triangles[:, 1], k=all_triangles[:, 2],
                color='rgb(210, 210, 210)', opacity=0.5, name='Part Body',
                showlegend=True, hoverinfo='skip'
            ))

        # 2. Setup-specific Locators
        for idx, step in enumerate(optimized_plan):
            axis = step['setup']

            # Extract points
            p_pts = step.get('PLF', {}).get('PLF_locators', [])
            s_pts = step.get('SLF', {}).get('SLF_locators', [])
            t_pts = step.get('TLF', {}).get('TLF_locators', [])

            # Primary Locators - RED
            if p_pts:
                pts = np.array(p_pts)
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers+text', name=f"Setup {axis} (Primary)",
                    legendgroup=f"group{axis}",
                    text=[f"P1", f"P2", f"P3"],
                    textposition="top center",
                    marker=dict(size=8, color='red', symbol='circle', line=dict(width=1, color='white'))
                ))

            # Secondary Locators - BLUE
            if s_pts:
                pts = np.array(s_pts)
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers+text', name=f"Setup {axis} (Secondary)",
                    legendgroup=f"group{axis}",
                    showlegend=False,
                    text=[f"S1", f"S2"],
                    textposition="top center",
                    marker=dict(size=8, color='blue', symbol='circle', line=dict(width=1, color='white'))
                ))

            # Tertiary Locators - GREEN
            if t_pts:
                pts = np.array(t_pts)
                fig.add_trace(go.Scatter3d(
                    x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                    mode='markers+text', name=f"Setup {axis} (Tertiary)",
                    legendgroup=f"group{axis}",
                    showlegend=False,
                    text=[f"T1"],
                    textposition="top center",
                    marker=dict(size=8, color='green', symbol='circle', line=dict(width=1, color='white'))
                ))

        # 3. Restored Diamond CoG
        cog = self.get_part_cog()
        fig.add_trace(go.Scatter3d(
            x=[cog[0]], y=[cog[1]], z=[cog[2]],
            mode='markers', name='Part CoG',
            marker=dict(size=12, color='purple', symbol='diamond')
        ))

        fig.update_layout(
            title="3-2-1 Workholding: Red(P), Blue(S), Green(T)",
            scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z", aspectmode="data"),
            legend=dict(itemsizing='constant', title_text="Click to Toggle Setups")
        )
        fig.show()