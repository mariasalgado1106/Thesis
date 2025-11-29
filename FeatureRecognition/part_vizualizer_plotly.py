import plotly.graph_objects as go
import plotly.io as pio
from geometry_analysis import load_step_file, analyze_shape
from aag_builder import AAGBuilder_3D

class Part_Visualizer:
    def __init__(self, builder: AAGBuilder_3D):
        self.builder = builder
        self.shape = builder.shape
        self.face_data_list = builder.face_data_list
        self.edge_data_list = builder.edge_data_list
        self.colors_rgb = builder.colors_rgb

        # mesh from builder
        self.vertices = getattr(builder, "vertices", None)
        self.triangles = getattr(builder, "triangles", None)

    def add_mesh_trace(self, fig, opacity=0.2, name="3D Model"):
        if self.vertices is None or self.triangles is None:
            return

        fig.add_trace(go.Mesh3d(
            x=[v[0] for v in self.vertices],
            y=[v[1] for v in self.vertices],
            z=[v[2] for v in self.vertices],
            i=[t[0] for t in self.triangles],
            j=[t[1] for t in self.triangles],
            k=[t[2] for t in self.triangles],
            color='lightblue',
            opacity=opacity,
            name=name,
            flatshading=True,
        ))


    def visualize_numbered_faces(self, node_size=10, title="Numbered faces"):

        face_idx = [f['index'] for f in self.face_data_list]
        face_types = [f['type'] for f in self.face_data_list]
        centers = [f['face_center'] for f in self.face_data_list]

        face_colors = {
            'Plane': self.colors_rgb['geo_plane'],
            'Cylinder': self.colors_rgb['geo_cylinder'],
            'Other': self.colors_rgb['geo_other']
        }

        fig = go.Figure()
        self.add_mesh_trace(fig,0.2)

        # one trace per type
        for ftype in sorted(set(face_types)):
            indices = [f['index'] for f in self.face_data_list if f['type'] == ftype]

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

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X",
                yaxis_title="Y",
                zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            legend=dict(
                yanchor="top", y=0.99,
                xanchor="left", x=0.01
            ),
            width=900,
            height=700
        )
        fig.show()

    def visualize_geometric_edges(self, title="Real edges by convexity"):
        fig = go.Figure()
        self.add_mesh_trace(fig, 0.2)

        # group by classification
        edge_groups = {
            "Convex": {"xs": [], "ys": [], "zs": [], "name": "Convex edges", "color": self.colors_rgb["edge_convex"],
                       "width": 4},
            "Concave": {"xs": [], "ys": [], "zs": [], "name": "Concave edges", "color": self.colors_rgb["edge_concave"],
                        "width": 4},
            "Tangent": {"xs": [], "ys": [], "zs": [], "name": "Tangent edges", "color": self.colors_rgb["edge_tangent"],
                        "width": 3},
            "Unknown": {"xs": [], "ys": [], "zs": [], "name": "Unknown edges", "color": (0.5, 0.5, 0.5), "width": 2},
        }

        for e in self.edge_data_list:
            cls = e["classification"][0] if e["classification"] else "Unknown"
            group = edge_groups.get(cls, edge_groups["Unknown"])

            points = e.get("points")
            if points is None or len(points) == 0:
                continue

            xs = points[:, 0].tolist()
            ys = points[:, 1].tolist()
            zs = points[:, 2].tolist()

            group["xs"].extend(xs + [None])
            group["ys"].extend(ys + [None])
            group["zs"].extend(zs + [None])

        # add one trace per group
        for grp in edge_groups.values():
            if not grp["xs"]:
                continue
            r, g, b = grp["color"]
            color_str = f"rgb({int(r * 255)},{int(g * 255)},{int(b * 255)})"
            fig.add_trace(go.Scatter3d(
                x=grp["xs"], y=grp["ys"], z=grp["zs"],
                mode="lines",
                line=dict(color=color_str, width=grp["width"]),
                name=grp["name"],
                hoverinfo="none"
            ))

        fig.update_layout(
            title=title,
            scene=dict(
                xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
                aspectmode="data",
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
            ),
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
            width=1000, height=800,
        )
        fig.show()
