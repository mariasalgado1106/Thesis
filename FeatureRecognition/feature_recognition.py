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
        self.subG_together = self.aag.build_aag_subgraph()
        self.colors_rgb = self.aag.colors_rgb

        # library
        self.lib = FeatureLibrary()
        self.blind_pattern = self.lib.get('feat_hole_blind')
        self.through_pattern = self.lib.get('feat_hole_through')

        self.matches: List[Dict] = []

    def identify_features(self) -> List[Dict]:
        feature_candidates = (self.subG_together.subgraph(c).copy()
            for c in nx.connected_components(self.subG_together))

        self.matches = []

        for candidate in feature_candidates:
            candidate_nodes = list(candidate.nodes())

            for name, pattern in self.lib.features.items():
                if nx.is_isomorphic(candidate, pattern): #compare candidate with the library pattern
                    self.matches.append({
                        'feature_type': name,
                        'node_indices': candidate_nodes, #store the faces that make the feature
                    })

        return self.matches

    def visualize_features_3d(self, show_mesh: bool = True, mesh_opacity: float = 0.6,
            node_size: int = 12,):
        import plotly.graph_objects as go

        fig = go.Figure()

        #associate faces with feature
        face_feature_type = {}  # face_idx -> feature_type
        for match in self.matches:
            ftype = match["feature_type"]
            for face_idx in match["node_indices"]:
                face_feature_type[face_idx] = ftype

        # mesh colored by feature
        if hasattr(self.aag, "vertices") and hasattr(self.aag, "triangles") and show_mesh:
            verts = self.aag.vertices
            tris = self.aag.triangles
            face_data_list = self.aag.face_data_list

            # You must have this: len == len(tris), each entry a face index
            triangle_face_index = self.aag.triangle_face_index

            facecolors = []
            for t_idx, tri in enumerate(tris):
                face_idx = triangle_face_index[t_idx]

                # Check if this face is part of a detected feature
                ftype = face_feature_type.get(face_idx)

                if ftype is not None:
                    # Color by feature (feat_*)
                    r, g, b = self.colors_rgb.get(ftype, self.colors_rgb["feat_other"])
                else:
                    # Fallback: color by geometry type
                    geom_type = face_data_list[face_idx]["type"]  # 'Plane', 'Cylinder', ...
                    if geom_type == "Plane":
                        r, g, b = self.colors_rgb["geo_plane"]
                    elif geom_type == "Cylinder":
                        r, g, b = self.colors_rgb["geo_cylinder"]
                    else:
                        r, g, b = self.colors_rgb["geo_other"]

                facecolors.append(f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})")

            fig.add_trace(go.Mesh3d(
                x=[v[0] for v in verts],
                y=[v[1] for v in verts],
                z=[v[2] for v in verts],
                i=[t[0] for t in tris],
                j=[t[1] for t in tris],
                k=[t[2] for t in tris],
                facecolor=facecolors,  # <--- per-triangle colors
                opacity=mesh_opacity,
                name="3D Model with features",
                flatshading=True,
            ))

        fig.update_layout(
            title="Detected features on 3D model",
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5)),
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=1200,
            height=900,
        )

        fig.show()
