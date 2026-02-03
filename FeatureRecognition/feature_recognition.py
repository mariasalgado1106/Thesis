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

            for name, pattern in self.lib.features.items():
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
                    print(f"NO match")

        print(f"\n=== FINAL RESULTS ===")
        print(f"Total matches: {len(self.matches)}")
        for match in self.matches:
            print(f"  {match['feature_type']}: nodes {match['node_indices']}")

        return self.matches

    def visualize_features_3d(self, show_mesh=True, mesh_opacity=0.7, node_size=15,
                                  show_face_centers=True, show_edges=True):
        import plotly.graph_objects as go
        import numpy as np

        fig = go.Figure()

        # Create a mapping: face_index -> feature info
        face_to_feature = {}
        for match in self.matches:
            feature_type = match['feature_type']
            for face_idx in match['node_indices']:
                face_to_feature[face_idx] = feature_type

        # Define feature colors
        feature_colors = self.colors_rgb

        # 1. Visualize face meshes colored by feature
        if show_mesh:
            # Group faces by feature type
            feature_groups = {}
            for face_data in self.face_data_list:
                face_idx = face_data['index']

                # Skip stock faces
                if face_data['stock_face'] == 'Yes':
                    continue

                # Determine feature membership
                if face_idx in face_to_feature:
                    feature = face_to_feature[face_idx]
                else:
                    feature = 'unrecognized'

                if feature not in feature_groups:
                    feature_groups[feature] = []
                feature_groups[feature].append(face_data)

            # Create one mesh trace per feature type
            for feature, faces in feature_groups.items():
                # Collect all vertices and triangles for this feature
                all_vertices = []
                all_triangles = []
                vertex_offset = 0

                for face_data in faces:
                    vertices = face_data.get('mesh_vertices', [])
                    triangles = face_data.get('mesh_triangles', [])

                    if not vertices or not triangles:
                        continue

                    # Add vertices
                    all_vertices.extend(vertices)

                    # Add triangles with offset
                    for tri in triangles:
                        all_triangles.append([
                            tri[0] + vertex_offset,
                            tri[1] + vertex_offset,
                            tri[2] + vertex_offset
                        ])

                    vertex_offset += len(vertices)

                if not all_vertices:
                    continue

                # Get color for this feature
                r, g, b = feature_colors.get(feature, (0.7, 0.7, 0.7))
                color_str = f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'

                # Create mesh trace
                fig.add_trace(go.Mesh3d(
                    x=[v[0] for v in all_vertices],
                    y=[v[1] for v in all_vertices],
                    z=[v[2] for v in all_vertices],
                    i=[t[0] for t in all_triangles],
                    j=[t[1] for t in all_triangles],
                    k=[t[2] for t in all_triangles],
                    color=color_str,
                    opacity=mesh_opacity,
                    name=f'{feature} ({len(faces)} faces)',
                    flatshading=False,
                    lighting=dict(ambient=0.5, diffuse=0.8, specular=0.2),
                    lightposition=dict(x=100, y=200, z=300)
                ))

            # 2. Face centers with labels
            if show_face_centers:
                centers = [f['face_center'] for f in self.face_data_list]

                # Define the name mapping once, outside the loop
                feature_name_map = {
                    'feat_hole_blind': 'Blind Hole',
                    'feat_hole_through': 'Through Hole'
                }

                # Single Loop: Iterate through features once
                for feature, color_rgb in feature_colors.items():
                    if feature == 'unrecognized':
                        continue

                    # Find faces belonging to this feature
                    indices = [
                        idx for idx in face_to_feature.keys()
                        if face_to_feature[idx] == feature
                           and self.face_data_list[idx]['stock_face'] != 'Yes'
                    ]

                    if not indices:
                        continue

                    # Get coordinates
                    xs = [centers[i][0] for i in indices]
                    ys = [centers[i][1] for i in indices]
                    zs = [centers[i][2] for i in indices]

                    # Determine color and name
                    r, g, b = color_rgb
                    color_str = f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'

                    # Get the pretty name (default to raw name if not in map)
                    display_name = feature_name_map.get(feature, feature)

                    fig.add_trace(go.Scatter3d(
                        x=xs, y=ys, z=zs,
                        mode='markers+text',
                        marker=dict(
                            size=node_size,
                            color=color_str,
                            line=dict(width=2, color='black')
                        ),
                        text=[str(i) for i in indices],
                        textposition='middle center',
                        textfont=dict(size=8, color='white'),
                        name=f'{display_name} Centers',
                        hovertemplate=(
                                "Face %{text}<br>Feature: " + display_name +
                                "<br>Center: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                        )
                    ))

            # 3. Edges (Geometric) - COMPLETELY SEPARATE BLOCK
            if show_edges:
                edge_groups = {
                    'Concave': {
                        'x': [], 'y': [], 'z': [],
                        'color': self.colors_rgb.get('edge_concave', (1, 0, 0)),
                        'width': 5
                    },
                    'Tangent': {
                        'x': [], 'y': [], 'z': [],
                        'color': self.colors_rgb.get('edge_tangent', (0, 0, 1)),
                        'width': 3
                    }
                }

                for edge in self.edge_data_list:
                    if not edge.get('classification'):
                        continue

                    edge_type = edge['classification'][0]

                    if edge_type in edge_groups:
                        points = edge['points']
                        # Add points separated by None to create discontinuous lines
                        edge_groups[edge_type]['x'].extend([p[0] for p in points] + [None])
                        edge_groups[edge_type]['y'].extend([p[1] for p in points] + [None])
                        edge_groups[edge_type]['z'].extend([p[2] for p in points] + [None])

                for name, group in edge_groups.items():
                    if not group['x']:
                        continue

                    r, g, b = group['color']
                    color_str = f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})'

                    fig.add_trace(go.Scatter3d(
                        x=group['x'], y=group['y'], z=group['z'],
                        mode='lines',
                        line=dict(color=color_str, width=group['width']),
                        name=f"{name} Edges"
                    ))

            # 4. Final layout
            fig.update_layout(
                title="Feature Recognition Visualization",
                scene=dict(
                    xaxis_title="X",
                    yaxis_title="Y",
                    zaxis_title="Z",
                    aspectmode="data",
                    camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
                ),
                legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
                width=1200,
                height=900
            )
            fig.show()