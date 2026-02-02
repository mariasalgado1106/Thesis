from typing import Dict, List, Tuple
import networkx as nx
import plotly.graph_objects as go
from aag_builder import AAGBuilder_2D, AAGBuilder_3D


class FeatureLibrary:
    def __init__(self):
        self.features: Dict[str, nx.Graph] = {}
        self.build_library()

    def build_library(self):
        # Blind hole: cylinder wall + plane bottom, concave edge
        G_blind = nx.Graph()
        G_blind.add_node(
            0,
            face_type='Cylinder',
            geometry='Cylinder',
            adjacent_faces=[1],
            stock_face='No'
        )
        G_blind.add_node(
            1,
            face_type='Plane',
            geometry='Plane',
            adjacent_faces=[0],
            stock_face='No'
        )
        G_blind.add_edge(0, 1, edge_type='concave')
        self.features['feat_hole_blind'] = G_blind

        # Through hole: 1 cylinder wall
        G_through = nx.Graph()
        G_through.add_node(
            0,
            face_type='Cylinder',
            geometry='Cylinder',
            adjacent_faces=[],
            stock_face='No'
        )
        self.features['feat_hole_through'] = G_through

    def get(self, name: str) -> nx.Graph:
        """Return a copy of the pattern graph so you don't mutate the library."""
        return self.features[name].copy()


class FeatureRecognition:
    def __init__(self, my_shape):
        self.aag = AAGBuilder_2D(my_shape)
        self.subgraphs_info = self.aag.analyse_subgraphs()
        self.colors_rgb = self.aag.colors_rgb

        # library
        self.lib = FeatureLibrary()
        self.blind_pattern = self.lib.get('feat_hole_blind')
        self.through_pattern = self.lib.get('feat_hole_through')

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


    def visualize_3d_feat(self, show_mesh=True, mesh_opacity=0.2, node_size=10):
        import plotly.graph_objects as go
        from collections import Counter

        fig = go.Figure()

        # 1. Mesh visualization
        if hasattr(self, "vertices") and hasattr(self, "triangles") and show_mesh:
            fig.add_trace(go.Mesh3d(
                x=[v[0] for v in self.vertices],
                y=[v[1] for v in self.vertices],
                z=[v[2] for v in self.vertices],
                i=[t[0] for t in self.triangles],
                j=[t[1] for t in self.triangles],
                k=[t[2] for t in self.triangles],
                color='lightblue',
                opacity=mesh_opacity,
                name='3D Model',
                flatshading=True,
            ))

        # 2. Face centers colored by face type
        face_types = [f['type'] for f in self.face_data_list]
        centers = [f['face_center'] for f in self.face_data_list]
        stock_face = [f['stock_face'] for f in self.face_data_list]

        face_colors = {
            'Plane': self.colors_rgb['geo_plane'],
            'Cylinder': self.colors_rgb['geo_cylinder'],
            'Other': self.colors_rgb['geo_other']
        }

        # one trace per type
        for ftype in sorted(set(face_types)):
            indices = [ #only non-stock faces
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
                marker=dict(
                    size=node_size,
                    color=color_str,
                    line=dict(width=2, color='black')
                ),
                text=[str(i) for i in indices],
                textposition='middle center',
                textfont=dict(size=8, color='black'),
                name=f'{ftype} faces ({len(indices)})',
                hovertemplate=(
                        "Face %{text}<br>Type: " + ftype +
                        "<br>Center: (%{x:.2f}, %{y:.2f}, %{z:.2f})<extra></extra>"
                )
            ))

        # 3. Edge links / AAG edges using self.colors_rgb
        edge_groups = {
            'Convex': {
                'edges': [],
                'color': self.colors_rgb['edge_convex'],
                'width': 4,
                'name': 'Convex edges'
            },
            'Concave': {
                'edges': [],
                'color': self.colors_rgb['edge_concave'],
                'width': 4,
                'name': 'Concave edges'
            },
            'Tangent': {
                'edges': [],
                'color': self.colors_rgb['edge_tangent'],
                'width': 3,
                'name': 'Tangent edges'
            },
            'Unknown': {
                'edges': [],
                'color': (0.5, 0.5, 0.5),  # grey
                'width': 2,
                'name': 'Unknown edges'
            }
        }

        drawn_pairs = set()
        for edge in self.edge_data_list:
            for pair in edge['faces_of_edge']:
                i, j = pair
                if i == j:
                    continue
                idx_pair = tuple(sorted((i, j)))
                if idx_pair in drawn_pairs:
                    continue
                drawn_pairs.add(idx_pair)

                # choose a classification string
                classification = 'Unknown'
                if edge['classification']:
                    # pick first or change logic if needed
                    classification = edge['classification'][0]

                if classification == 'Convex':
                    continue

                group = edge_groups.get(classification, edge_groups['Unknown'])
                group['edges'].append((i, j, edge))

        for group in edge_groups.values():
            if not group['edges']:
                continue
            x_line, y_line, z_line = [], [], []
            hover_text = []
            for (f1, f2, e_info) in group['edges']:
                x1, y1, z1 = centers[f1]
                x2, y2, z2 = centers[f2]
                x_line.extend([x1, x2, None])
                y_line.extend([y1, y2, None])
                z_line.extend([z1, z2, None])
                hover_label = (
                    f"Edge {e_info['index']}<br>Faces: {f1} ↔ {f2}<br>"
                    f"Classification: {e_info['classification']}<br>"
                    f"Length: {e_info['edge_length']:.2f}"
                )
                hover_text.extend([hover_label, hover_label, None])

            r, g, b = group['color']
            fig.add_trace(go.Scatter3d(
                x=x_line, y=y_line, z=z_line, mode='lines',
                line=dict(
                    color=f'rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})',
                    width=group['width']
                ),
                name=f"{group['name']} ({len(group['edges'])})",
                hovertemplate="%{text}<extra></extra>",
                text=hover_text
            ))

        # 4. Features


        # 5. Final things
        fig.update_layout(
            title="AAG Graph from 3D Model" + (" (no convex edges and stock faces)" if hide_convex else ""),
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