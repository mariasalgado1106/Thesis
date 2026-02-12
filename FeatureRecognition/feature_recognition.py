from typing import Dict, List, Tuple
import networkx as nx
import plotly.graph_objects as go
from aag_builder import AAGBuilder_2D, AAGBuilder_3D
from geometry_analysis import load_step_file, analyze_shape
from part_vizualizer_plotly import Part_Visualizer


class FeatureLibrary:
    def __init__(self):
        self.features: Dict[str, nx.Graph] = {}
        self.build_library()

    def build_library(self):
        # Blind hole: cylinder wall + plane bottom, concave edge
        G_hole_blind = nx.Graph()
        G_hole_blind.add_node(
            0,
            face_type='Cylinder',
            geometry='Cylinder',
            adjacent_faces=[1],
            stock_face='No'
        )
        G_hole_blind.add_node(
            1,
            face_type='Plane',
            geometry='Plane',
            adjacent_faces=[0],
            stock_face='No'
        )
        G_hole_blind.add_edge(0, 1, edge_type='concave')
        self.features['feat_hole_blind'] = G_hole_blind

        # Through hole: 1 cylinder wall
        G_hole_through = nx.Graph()
        G_hole_through.add_node(
            0,
            face_type='Cylinder',
            geometry='Cylinder',
            adjacent_faces=[],
            stock_face='No'
        )
        self.features['feat_hole_through'] = G_hole_through

        # Blind Pocket: 5 nodes, 1 base connected with concave to all the other 4 and
        # the other 4 creating a "loop" of concave connection
        # all nodes are planes
        G_pocket_blind = nx.Graph()

        G_pocket_blind.add_node(0, face_type='Plane', geometry='Plane',
                                adjacent_faces=[1, 2, 3, 4], stock_face='No')

        G_pocket_blind.add_node(1, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 2, 4], stock_face='No')

        G_pocket_blind.add_node(2, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 1, 3], stock_face='No')

        G_pocket_blind.add_node(3, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 2, 4], stock_face='No')

        G_pocket_blind.add_node(4, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 1, 3], stock_face='No')

        # 1. Connect Node 0 to all 4 walls
        G_pocket_blind.add_edge(0, 1, edge_type='concave')
        G_pocket_blind.add_edge(0, 2, edge_type='concave')
        G_pocket_blind.add_edge(0, 3, edge_type='concave')
        G_pocket_blind.add_edge(0, 4, edge_type='concave')

        # 2. Connect the walls to each other (The cycle 1-2-3-4-1)
        G_pocket_blind.add_edge(1, 2, edge_type='concave')
        G_pocket_blind.add_edge(2, 3, edge_type='concave')
        G_pocket_blind.add_edge(3, 4, edge_type='concave')
        G_pocket_blind.add_edge(4, 1, edge_type='concave')
        self.features['feat_pocket_blind'] = G_pocket_blind


        # Through Pocket: 4 nodes creating a "loop" of concave connection
        # all nodes are planes
        G_pocket_through = nx.Graph()
        G_pocket_through.add_node(1, face_type='Plane', geometry='Plane',
                                adjacent_faces=[2, 4], stock_face='No')

        G_pocket_through.add_node(2, face_type='Plane', geometry='Plane',
                                adjacent_faces=[1, 3], stock_face='No')

        G_pocket_through.add_node(3, face_type='Plane', geometry='Plane',
                                adjacent_faces=[2, 4], stock_face='No')

        G_pocket_through.add_node(4, face_type='Plane', geometry='Plane',
                                adjacent_faces=[1, 3], stock_face='No')

        G_pocket_through.add_edge(1, 2, edge_type='concave')
        G_pocket_through.add_edge(2, 3, edge_type='concave')
        G_pocket_through.add_edge(3, 4, edge_type='concave')
        G_pocket_through.add_edge(4, 1, edge_type='concave')
        self.features['feat_pocket_through'] = G_pocket_through


        # Blind Slot: 4 nodes, 0 and 1 connected and 2 and 3 connected to both 0 and 1
        # all nodes are planes
        G_slot_blind = nx.Graph()
        G_slot_blind.add_node(0, face_type='Plane', geometry='Plane',
                                adjacent_faces=[1, 2, 3], stock_face='No')

        G_slot_blind.add_node(1, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 2, 3], stock_face='No')

        G_slot_blind.add_node(2, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 1], stock_face='No')

        G_slot_blind.add_node(3, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 1], stock_face='No')

        G_slot_blind.add_edge(0, 1, edge_type='concave')
        G_slot_blind.add_edge(0, 2, edge_type='concave')
        G_slot_blind.add_edge(0, 3, edge_type='concave')
        G_slot_blind.add_edge(1, 2, edge_type='concave')
        G_slot_blind.add_edge(1, 3, edge_type='concave')
        self.features['feat_slot_blind'] = G_slot_blind

        # Through Slot: 3 nodes, 0 connected to both 1 and 2
        # all nodes are planes
        G_slot_through = nx.Graph()
        G_slot_through.add_node(0, face_type='Plane', geometry='Plane',
                                adjacent_faces=[1, 2], stock_face='No')

        G_slot_through.add_node(1, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0], stock_face='No')

        G_slot_through.add_node(2, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0], stock_face='No')

        G_slot_through.add_edge(0, 1, edge_type='concave')
        G_slot_through.add_edge(0, 2, edge_type='concave')
        self.features['feat_slot_through'] = G_slot_through

        # Blind Step: 3 nodes connected in "loop"
        # all nodes are planes
        G_step_blind = nx.Graph()
        G_step_blind.add_node(0, face_type='Plane', geometry='Plane',
                                adjacent_faces=[1, 2], stock_face='No')

        G_step_blind.add_node(1, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 2], stock_face='No')

        G_step_blind.add_node(2, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0, 1], stock_face='No')

        G_step_blind.add_edge(0, 1, edge_type='concave')
        G_step_blind.add_edge(0, 2, edge_type='concave')
        G_step_blind.add_edge(1, 2, edge_type='concave')
        self.features['feat_step_blind'] = G_step_blind

        # Through Step: 2 connected nodes
        # all nodes are planes
        G_step_through = nx.Graph()
        G_step_through.add_node(0, face_type='Plane', geometry='Plane',
                                adjacent_faces=[1], stock_face='No')

        G_step_through.add_node(1, face_type='Plane', geometry='Plane',
                                adjacent_faces=[0], stock_face='No')

        G_step_through.add_edge(0, 1, edge_type='concave')
        self.features['feat_step_through'] = G_step_through

    def get(self, name: str) -> nx.Graph:
        """Return a copy of the pattern graph so you don't mutate the library."""
        return self.features[name].copy()


class FeatureRecognition:
    def __init__(self, my_shape):
        self.aag = AAGBuilder_2D(my_shape)
        self.subgraphs_info = self.aag.analyse_subgraphs()
        self.colors_rgb = self.aag.colors_rgb
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)

        # library
        self.lib = FeatureLibrary()


        self.matches: List[Dict] = []

    # Free Form Pocket: 1 base node connected to n-1 nodes, that are all connected in a loop
    # all nodes are planes
    def build_free_form_pocket(self, n):
        #BLIND
        G_pocket_freeform_blind = nx.Graph()
        # base node
        G_pocket_freeform_blind.add_node(0, face_type='Plane', geometry='Plane', stock_face='No')
        # walls 1 -> n-1
        for i in range(1, n):
            G_pocket_freeform_blind.add_node(i, face_type='Plane', geometry='Plane', stock_face='No')
            # Every wall connects to the base
            G_pocket_freeform_blind.add_edge(0, i, edge_type='concave')

            # Connect to the previous wall
            if i > 1:
                G_pocket_freeform_blind.add_edge(i, i - 1, edge_type='concave')

        # last wall must connect back to the first wall (node 1)
        if n > 2:
            G_pocket_freeform_blind.add_edge(n - 1, 1, edge_type='concave')

        #THROUGH -> NO BASE NODE
        G_pocket_freeform_through = nx.Graph()
        # walls 1 -> n-1
        for i in range(1, n):
            G_pocket_freeform_through.add_node(i, face_type='Plane', geometry='Plane', stock_face='No')
            # Connect to the previous wall
            if i > 1:
                G_pocket_freeform_through.add_edge(i, i - 1, edge_type='concave')
        # last wall must connect back to the first wall (node 1)
        if n > 2:
            G_pocket_freeform_through.add_edge(n - 1, 1, edge_type='concave')

        return G_pocket_freeform_blind, G_pocket_freeform_through

    def is_conjoined_pocket(self, candidate_graph, candidate_nodes):
        n = len(candidate_nodes)
        if n < 5: return False

        # 1. Identify the Base Node (highest degree)
        # In a conjoined pocket, the floor is connected to all walls.
        degrees = dict(candidate_graph.degree())
        base_node = max(degrees, key=degrees.get)
        if degrees[base_node] != n - 1:
            return False

        # 2. Check Edge Count Rule
        # We solve for 'p' (number of pockets): p = 2(n-1) - Edges
        num_edges = candidate_graph.number_of_edges()
        p = (2 * (n - 1)) - num_edges

        # If p > 1, it's likely a conjoined pocket
        if p >= 2:
            return True, p
        return False, 0

    # Free Form Slot: 1 base node connected to n-1 nodes
    # all nodes are planes
    def build_free_form_slot(self, n):
        # THROUGH
        G_slot_freeform_through = nx.Graph()
        # base node
        G_slot_freeform_through.add_node(0, face_type='Plane', geometry='Plane', stock_face='No')
        # walls 1 -> n-1
        for i in range(1, n):
            G_slot_freeform_through.add_node(i, face_type='Plane', geometry='Plane', stock_face='No')
            # Every wall connects to the base
            G_slot_freeform_through.add_edge(0, i, edge_type='concave')

        return G_slot_freeform_through



    def identify_features(self) -> List[Dict]:

        print(f"\n=== IDENTIFY FEATURES ===")
        print(f"Total connected components: {len(self.subgraphs_info)}")

        feature_candidates = self.subgraphs_info
        self.matches = []

        for candidate_info in feature_candidates:  # Renamed for clarity
            candidate_idx = candidate_info['subgraph_idx']
            candidate_graph = candidate_info['subgraph']
            candidate_nodes = candidate_info['nodes']
            n_candidate_concave = candidate_info['n_concave']
            matched = False

            node_match = nx.algorithms.isomorphism.categorical_node_match('face_type', None)
            edge_match = nx.algorithms.isomorphism.categorical_edge_match('edge_type', None)

            for name, pattern in self.lib.features.items():
                if nx.is_isomorphic(candidate_graph, pattern, node_match=node_match, edge_match=edge_match):
                    self.matches.append({'feature_type': name, 'node_indices': candidate_nodes})
                    matched = True
                    print(f"MATCH! Feature {self.matches[-1]['feature_type']} found")
                    break

            # If no library match, try the other ones
            if not matched:
                n_nodes = len(candidate_nodes)
                G_pocket_freeform_blind, G_pocket_freeform_through = self.build_free_form_pocket(n_nodes)

                if nx.is_isomorphic(candidate_graph, G_pocket_freeform_blind, node_match=node_match,
                                            edge_match=edge_match):
                    self.matches.append({
                        'feature_type': f'feat_pocket_blind',
                        'node_indices': candidate_nodes,
                    })
                    matched = True
                    print(f"MATCH! Free-form blind pocket with {n_nodes} faces found.")


                elif nx.is_isomorphic(candidate_graph, G_pocket_freeform_through, node_match=node_match,
                edge_match=edge_match):
                    self.matches.append({
                        'feature_type': f'feat_pocket_through',
                        'node_indices': candidate_nodes,
                    })
                    matched = True
                    print(f"MATCH! Free-form through pocket with {n_nodes} faces found.")


            if not matched:
                n_nodes = len(candidate_nodes)
                G_slot_freeform_through= self.build_free_form_slot(n_nodes)

                if nx.is_isomorphic(candidate_graph, G_slot_freeform_through, node_match=node_match,
                                            edge_match=edge_match):
                    self.matches.append({
                        'feature_type': f'feat_slot_through',
                        'node_indices': candidate_nodes,
                    })
                    matched = True
                    print(f"MATCH! Free-form through slot with {n_nodes} faces found.")



            if not matched:
                # Try the Rule-Based approach for Conjoined Pockets
                is_conjoined, num_pockets = self.is_conjoined_pocket(candidate_graph, candidate_nodes)

                if is_conjoined:
                    self.matches.append({
                        'feature_type': f'feat_pocket_blind',
                        'node_indices': candidate_nodes,
                        'num_sub_features': num_pockets
                    })
                    print(f"MATCH! Conjoined feature found with {num_pockets} pockets.")
                    matched = True


            if not matched:
                print(f"Candidate {candidate_idx} still unrecognized.")



            '''for name, pattern in self.lib.features.items():
                print(f"\nChecking if candidate {candidate_idx} matches {name}")
                print(f"  Candidate nodes: {len(candidate_nodes)}, Pattern nodes: {len(pattern.nodes())}")
                print(f"  Candidate edges: {n_candidate_concave}, Pattern edges: {len(pattern.edges())}")

                node_match = nx.algorithms.isomorphism.categorical_node_match('face_type', None)
                edge_match = nx.algorithms.isomorphism.categorical_edge_match('edge_type', None)

                # use graph here
                if nx.is_isomorphic(candidate_graph, pattern, node_match=node_match, edge_match=edge_match):
                    self.matches.append({
                        'feature_type': name,
                        'node_indices': candidate_nodes,
                    })
                    print(f"MATCH! Feature {self.matches[-1]['feature_type']} found")
                    break  # Stop checking other patterns for this candidate
                else:
                    G_pocket_freeform = self.build_free_form_pocket(len(candidate_nodes))
                    if nx.is_isomorphic(candidate_graph, G_pocket_freeform, node_match=node_match, edge_match=edge_match):
                        self.matches.append({
                            'feature_type': "Free Form Pocket",
                            'node_indices': candidate_nodes,
                        })
                        print(f"MATCH! Feature {self.matches[-1]['feature_type']} found")
                else:
                    print(f"NO match")'''

        print(f"\n=== FINAL RESULTS ===")
        print(f"Total matches: {len(self.matches)}")
        for match in self.matches:
            print(f"  {match['feature_type']}: nodes {match['node_indices']}")

        return self.matches

    def visualize_features_3d(self, show_mesh=True, mesh_opacity=0.7,
                              show_face_centers=True, show_edges=True):
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        feature_name_map = {
            'feat_hole_blind': 'Blind Hole',
            'feat_hole_through': 'Through Hole',
            'feat_pocket_blind': 'Blind Pocket',
            'feat_pocket_through': 'Through Pocket',
            'feat_slot_blind': 'Blind Slot',
            'feat_slot_through': 'Through Slot',
            'feat_step_blind': 'Blind Step',
            'feat_step_through': 'Through Step',
            'unrecognized': 'Unrecognized Face',
            'stock': 'Part Body'
        }

        face_to_feature = {}
        for match in self.matches:
            feature_type = match['feature_type']
            for face_idx in match['node_indices']:
                face_to_feature[face_idx] = feature_type

        feature_colors = self.colors_rgb

        # Colored Features + mesh
        if show_mesh:
            feature_groups = {}
            for face_data in self.face_data_list:
                face_idx = face_data['index']

                if face_data['stock_face'] == 'Yes':
                    group_key = 'stock'
                elif face_idx in face_to_feature:
                    group_key = face_to_feature[face_idx]
                else:
                    group_key = 'unrecognized'

                if group_key not in feature_groups:
                    feature_groups[group_key] = []
                feature_groups[group_key].append(face_data)

            for group_key, faces in feature_groups.items():
                all_vertices = []
                all_triangles = []
                vertex_offset = 0

                for face_data in faces:
                    vertices = face_data.get('mesh_vertices', [])
                    triangles = face_data.get('mesh_triangles', [])

                    if not vertices or not triangles:
                        continue

                    all_vertices.extend(vertices)
                    for tri in triangles:
                        all_triangles.append([
                            tri[0] + vertex_offset,
                            tri[1] + vertex_offset,
                            tri[2] + vertex_offset
                        ])
                    vertex_offset += len(vertices)

                if not all_vertices:
                    continue

                if group_key == 'stock':
                    color_str = 'rgb(200, 200, 200)'
                    current_opacity = 0.1
                elif group_key == 'unrecognized':
                    color_str = 'rgb(100, 100, 100)'
                    current_opacity = 0.5
                else:
                    r, g, b = feature_colors.get(group_key, (0.7, 0.7, 0.7))
                    color_str = f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'
                    current_opacity = mesh_opacity

                display_name = feature_name_map.get(group_key, group_key)

                fig.add_trace(go.Mesh3d(
                    x=[v[0] for v in all_vertices],
                    y=[v[1] for v in all_vertices],
                    z=[v[2] for v in all_vertices],
                    i=[t[0] for t in all_triangles],
                    j=[t[1] for t in all_triangles],
                    k=[t[2] for t in all_triangles],
                    color=color_str,
                    opacity=current_opacity,
                    name=display_name,
                    showlegend=True,
                    flatshading=False,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                    lightposition=dict(x=100, y=200, z=300)
                ))

        # 2. FACE nodes
        if show_face_centers:
            face_types = [f['type'] for f in self.face_data_list]
            centers = [f['face_center'] for f in self.face_data_list]

            face_colors = {
                'Plane': self.colors_rgb.get('geo_plane', (0, 1, 0)),
                'Cylinder': self.colors_rgb.get('geo_cylinder', (1, 0, 0)),
                'Other': self.colors_rgb.get('geo_other', (0, 0, 1))
            }

            for ftype in sorted(set(face_types)):
                # only non-stock faces
                indices = [
                    f['index']
                    for f in self.face_data_list
                    if f['type'] == ftype and f['stock_face'] != "Yes"
                ]

                if not indices:
                    continue

                xs = [centers[i][0] for i in indices]
                ys = [centers[i][1] for i in indices]
                zs = [centers[i][2] for i in indices]

                r, g, b = face_colors.get(ftype, (0.5, 0.5, 0.5))
                color_str = f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'

                fig.add_trace(go.Scatter3d(
                    x=xs, y=ys, z=zs,
                    mode='markers+text',
                    marker=dict(size=5, color=color_str, line=dict(width=1, color='black')),
                    text=[str(i) for i in indices],
                    name=f'{ftype} Nodes',
                    showlegend=False
                ))

        # 3. EDGES
        if show_edges:
            edge_groups = {
                'Convex': {
                    'x': [], 'y': [], 'z': [],
                    'color': self.colors_rgb.get('edge_convex', (0, 0, 1)),  # Blue-ish
                    'width': 4
                },
                'Concave': {
                    'x': [], 'y': [], 'z': [],
                    'color': self.colors_rgb.get('edge_concave', (1, 0, 0)),  # Red
                    'width': 4
                },
                'Tangent': {
                    'x': [], 'y': [], 'z': [],
                    'color': self.colors_rgb.get('edge_tangent', (0, 1, 0)),  # Green
                    'width': 3
                },
                'Unknown': {
                    'x': [], 'y': [], 'z': [],
                    'color': (0.5, 0.5, 0.5),
                    'width': 2
                }
            }

            for edge in self.edge_data_list:
                # Type
                etype = 'Unknown'
                if edge.get('classification'):
                    etype = edge['classification'][0]  # Take the first classification

                # Geometry Points
                points = edge.get('points', [])
                if len(points) == 0:
                    continue

                group = edge_groups.get(etype, edge_groups['Unknown'])

                group['x'].extend([p[0] for p in points] + [None])
                group['y'].extend([p[1] for p in points] + [None])
                group['z'].extend([p[2] for p in points] + [None])

            for name, group in edge_groups.items():
                if not group['x']:
                    continue

                r, g, b = group['color']
                color_str = f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'

                fig.add_trace(go.Scatter3d(
                    x=group['x'], y=group['y'], z=group['z'],
                    mode='lines',
                    line=dict(color=color_str, width=group['width']),
                    name=f"{name} Edges",
                    showlegend=True
                ))

        fig.update_layout(
            title="Feature Recognition Visualization",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data"
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=1200, height=900
        )
        fig.show()