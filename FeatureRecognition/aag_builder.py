import json
from collections import defaultdict, Counter

import numpy as np
import matplotlib.pyplot as plt
from jinja2.nodes import Continue
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import plotly.graph_objects as go
import plotly.io as pio
import networkx as nx
from typing import List, Dict, Any, Set, Tuple

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopAbs import TopAbs_FORWARD
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties, brepgprop_LinearProperties
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_Circle)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve

from geometry_analysis import load_step_file, analyze_shape




'''
#I ONLY NEED THIS IF I GET FROM OUTSIDE FILES, WHICH IM NOT USING
def edge_key(edge, tolerance=1e-6):
    try:
        curve_adapter = BRepAdaptor_Curve(edge)
        curve_type = curve_adapter.GetType()

        if curve_type == GeomAbs_Circle:
            circle = curve_adapter.Circle()
            center = circle.Location()
            radius = circle.Radius()
            center_key = (
                round(center.X() / tolerance) * tolerance,
                round(center.Y() / tolerance) * tolerance,
                round(center.Z() / tolerance) * tolerance,
                round(radius / tolerance) * tolerance
            )
            return ("circle",) + center_key
        else:
            vertices = list(TopologyExplorer(edge).vertices())
            if len(vertices) != 2:
                return None
            p1 = BRep_Tool.Pnt(vertices[0])
            p2 = BRep_Tool.Pnt(vertices[1])
            coords = sorted([
                (round(p1.X() / tolerance) * tolerance, round(p1.Y() / tolerance) * tolerance,
                 round(p1.Z() / tolerance) * tolerance),
                (round(p2.X() / tolerance) * tolerance, round(p2.Y() / tolerance) * tolerance,
                 round(p2.Z() / tolerance) * tolerance)
            ])
            return ("line",) + tuple(coords[0] + coords[1])
    except:
        return None
'''

#generate a triangulated mesh of the part for Plotly visualization
def mesh_shape_for_visualization(shape, linear_deflection=0.1):
    BRepMesh_IncrementalMesh(shape, linear_deflection)

#extract a renderable triangle mesh from the B‑rep
def extract_mesh_data(shape):
    vertices = []
    triangles = []
    vertex_count = 0
    topo = TopologyExplorer(shape)
    for face in topo.faces():
        loc = TopLoc_Location()
        triangulation = BRep_Tool.Triangulation(face, loc)
        if triangulation is None:
            continue
        transform = loc.Transformation()
        face_vertices = []
        for i in range(1, triangulation.NbNodes() + 1):
            pnt = triangulation.Node(i)
            pnt.Transform(transform)
            vertices.append([pnt.X(), pnt.Y(), pnt.Z()])
            face_vertices.append(vertex_count)
            vertex_count += 1
        for i in range(1, triangulation.NbTriangles() + 1):
            triangle = triangulation.Triangle(i)
            n1, n2, n3 = triangle.Get()
            if face.Orientation() == TopAbs_FORWARD:
                triangles.append([face_vertices[n1 - 1], face_vertices[n2 - 1], face_vertices[n3 - 1]])
            else:
                triangles.append([face_vertices[n1 - 1], face_vertices[n3 - 1], face_vertices[n2 - 1]])
    return vertices, triangles



class AAGBuilder_3D:
    def __init__(self, my_shape):
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)

        self.colors_rgb = {
            # EDGES
            "edge_concave": (1.0, 0.0, 0.0),  # Red
            "edge_convex": (0.0, 0.0, 1.0),  # Blue
            "edge_tangent": (0.0, 0.0, 0.0),  # Black

            # FEATURES
            "feat_hole_through": (0.0, 0.0, 0.45),  # Dark Blue
            "feat_hole_blind": (0, 0.3, 1),  # Light Blue
            "feat_pocket_through": (0.0, 0.39, 0.0),  # Dark Green
            "feat_pocket_blind": (0.2, 0.8, 0.2),  # Light Green
            "feat_slot_through": (1.0, 0.08, 0.58),  # Dark Pink
            "feat_slot_blind": (1.0, 0.41, 0.71),  # Hot Pink
            "feat_step": (1.0, 0.55, 0.0),  # Dark Orange
            "feat_other": (1.0, 0.2, 0.1),  # Light Orange

            # GEOMETRY
            "geo_plane": (0.75, 0.75, 0.75),  # Light Grey
            "geo_cylinder": (0.94, 0.98, 0.72),  # Pale Yellow
            "geo_other": (0.4, 0.4, 0.4),  # Darker Gray
        }

    def load_shape (self):
        mesh_shape_for_visualization(self.shape, linear_deflection=0.1)
        self.vertices, self.triangles = extract_mesh_data(self.shape)

    '''
    #ONLY NEED IF I LOAD A FILE AND WANT TO CLASSIFY/MAP THE EDGES
    def load_convexity_results(self):
        self.edge_classification = {}
        for edge_data in self.edge_data_list:
            edge = edge_data['edge']
            self.edge_classification[edge] = {
                'edge_idx': edge_data['index'],
                'edge' : edge_data['edge'],
                'classification': edge_data['classification'],
                'edge_geom': edge_data['edge_geom'],
                'edge_length': edge_data['edge_length']
            }
        print(f"Loaded convexity results for {len(self.edge_data_list)} edges")

    
    def build_consistent_edge_mapping(self): #link the edges from the file to the ones from the edge_data_list
        topo = TopologyExplorer(self.shape)
        self.edges = list(topo.edges()) #get the edges
        self.edge_id_map = {}
        self.convexity_to_current_map = {}

        for conv_id in self.edge_classification.keys():
            if conv_id < len(self.edges):
                edge = self.edges[conv_id]
                key = edge_key(edge)
                if key is not None:
                    self.edge_id_map[key] = conv_id
                    self.convexity_to_current_map[conv_id] = conv_id

        print(f"Direct mapping: {len(self.edge_id_map)} edges mapped")

        unmapped_conv = set(self.edge_classification.keys()) - set(self.convexity_to_current_map.keys())
        for conv_id in unmapped_conv:
            conv_data = self.edge_classification[conv_id]
            conv_length = conv_data['edge_length']
            for curr_id, edge in enumerate(self.edges):
                if curr_id in self.convexity_to_current_map.values():
                    continue
                try:
                    props = GProp_GProps()
                    brepgprop_LinearProperties(edge, props)
                    curr_length = props.Mass()
                    if abs(curr_length - conv_length) < 0.1:
                        key = edge_key(edge)
                        if key is not None and key not in self.edge_id_map:
                            self.edge_id_map[key] = conv_id
                            self.convexity_to_current_map[conv_id] = curr_id
                            break
                except:
                    continue

        final_unmapped = set(self.edge_classification.keys()) - set(self.convexity_to_current_map.keys())
        print(f"Final: {len(self.edge_id_map)} mapped, {len(final_unmapped)} unmapped")'''

    def visualize_3d_aag(self, show_mesh=True, mesh_opacity=0.2, node_size=10, hide_convex=False):
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

            if hide_convex:
                # keep only non‑stock faces
                indices = [
                    f['index']
                    for f in self.face_data_list
                    if f['type'] == ftype and f['stock_face'] != "Yes"
                ]
            else:
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

                if hide_convex and classification == 'Convex':
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

        print("\nFace type distribution:")
        for ftype, count in Counter(face_types).items():
            print(f"  {ftype}: {count}")
        edge_class_flat = []
        for edge in self.edge_data_list:
            edge_class_flat.extend(edge['classification'])
        print("\nEdge classification distribution:")
        for cls, count in Counter(edge_class_flat).items():
            print(f"  {cls}: {count}")
        print(f"\nTotal faces: {len(self.face_data_list)}")
        print(f"Total mesh edges: {len(self.edge_data_list)}")
        print(f"Graph/AAG links (drawn pairs): {len(drawn_pairs)}")










########################################
class AAGBuilder_2D:
    def __init__(self, my_shape):
        self.shape = my_shape
        self.all_faces, self.face_data_list, _ , _, _ = analyze_shape(self.shape)

        self.G = None
        self.subG = None
        self.subgraphs_info = []

        self.colors_rgb = {
            # EDGES
            "edge_concave": (1.0, 0.0, 0.0),  # Red
            "edge_convex": (0.0, 0.0, 1.0),  # Blue
            "edge_tangent": (0.0, 0.0, 0.0),  # Black

            # FEATURES
            "feat_hole_through": (0.0, 0.0, 0.45),  # Dark Blue
            "feat_hole_blind": (0, 0.3, 1),  # Light Blue
            "feat_pocket_through": (0.0, 0.39, 0.0),  # Dark Green
            "feat_pocket_blind": (0.2, 0.8, 0.2),  # Light Green
            "feat_slot_through": (1.0, 0.08, 0.58),  # Dark Pink
            "feat_slot_blind": (1.0, 0.41, 0.71),  # Hot Pink
            "feat_step": (1.0, 0.55, 0.0),  # Dark Orange
            "feat_other": (1.0, 0.2, 0.1),  # Light Orange

            # GEOMETRY
            "geo_plane": (0.96, 0.96, 0.96),  # Light Grey
            "geo_cylinder": (1.0, 0.98, 0.8),  # Pale Yellow
            "geo_other": (0.98, 0.94, 0.90),  # Beige
        }

    # 1. BUILD GRAPH AND SUBGRAPH
    def build_aag_graph(self): #graph with all edge types
        self.G = nx.Graph()
        all_faces, face_data_list, _, _, _ = analyze_shape(self.shape)

        for face_data in self.face_data_list:
            i = face_data["index"]
            self.G.add_node(i, face_type=face_data["type"], geometry=face_data["geom"],
                       adjacent_faces=face_data["adjacent_indices"], stock_face=face_data["stock_face"])

        # Add ALL edge types
        for face_data in self.face_data_list:
            current_face = face_data["index"]

            for adj_idx in face_data["convex_adjacent"]:
                if not self.G.has_edge(current_face, adj_idx):
                    self.G.add_edge(current_face, adj_idx, edge_type="convex")

            for adj_idx in face_data["concave_adjacent"]:
                if not self.G.has_edge(current_face, adj_idx):
                    self.G.add_edge(current_face, adj_idx, edge_type="concave")

            for adj_idx in face_data["tangent_adjacent"]:
                if not self.G.has_edge(current_face, adj_idx):
                    self.G.add_edge(current_face, adj_idx, edge_type="tangent")

        print(f"TRIAL Total nodes: {self.G.number_of_nodes()}")
        print(f"TRIAL Total edges (complete graph): {self.G.number_of_edges()}")

        convex_count = sum(1 for x, y, z in self.G.edges(data=True) if z.get('edge_type') == 'convex')
        print(f"TRIAL Total convex edges: {convex_count}")

        concave_count = sum(1 for x, y, z in self.G.edges(data=True) if z.get('edge_type') == 'concave')
        print(f"TRIAL Total concave edges: {concave_count}")

        tangent_count = sum(1 for x, y, z in self.G.edges(data=True) if z.get('edge_type') == 'tangent')
        print(f"TRIAL Total tangent edges: {tangent_count}")

        return self.G


    def build_aag_subgraph (self):
        if self.G is None:
            self.build_aag_graph()
        def filter_stock_faces(node):
            return self.G.nodes[node].get("stock_face") != "Yes"
        def filter_edge(n1, n2): #n1 and n2 are the 2 nodes
            return self.G[n1][n2].get("edge_type") != "convex" #if convex it returns false and removes
        self.subG = nx.subgraph_view(self.G, filter_node=filter_stock_faces, filter_edge=filter_edge)
        #this only created the view of the graph, not actually removing those nodes

        #make sure planar faces by themselves are removed
        nodes_to_keep = set()
        for component in nx.connected_components(self.subG):
            sg = self.subG.subgraph(component)
            # Keep if: has edges OR is a cylinder (potential through-hole)
            if sg.number_of_edges() > 0:
                nodes_to_keep.update(component)
            else:
                # Single isolated node - only keep if it's a cylinder
                node = list(component)[0]
                if self.G.nodes[node].get('face_type') == 'Cylinder':
                    nodes_to_keep.add(node)

        # Create final filtered subgraph
        self.subG = self.subG.subgraph(nodes_to_keep).copy()

        return self.subG

    # 2. ANALYSE SUBGRAPH FOR FR (connected faces)

    def analyse_subgraphs(self):
        if self.subG is None:
            self.build_aag_subgraph()
        subgraphs = list(nx.connected_components(self.subG))
        self.subgraphs_info = []

        for i, nodeset in enumerate(subgraphs):
            sg = self.subG.subgraph(nodeset).copy()
            nodes = list(sg.nodes())
            n_faces = len(nodes)
            face_types = [self.face_data_list[node]['type'] for node in nodes]
            n_concave = sum(1 for _, _, data in sg.edges(data=True) if data.get('edge_type') == 'concave')
            print(f"Subgraph {i}: faces={n_faces}, concave_edges={n_concave}")

            self.subgraphs_info.append({
                'subgraph_idx': i,
                'subgraph': sg,  # the actual graph object
                'nodes': nodes,
                'n_faces': n_faces,
                'n_concave': n_concave,
                'face_types': face_types
            })
        return self.subgraphs_info


    # 3. VISUALIZE GRAPHS
    def visualize_2d_aag (self):
        if self.G is None:
            self.build_aag_graph()
        if self.subG is None:
            self.build_aag_subgraph()

        nodes_positions = nx.spring_layout(self.G, seed=42) #aesthetic way of representing the nodes
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) #side by side plots

        # 1: Create Graphs
        # 1.1: Colors by geometry
        node_colors_G = []
        for n in self.G.nodes:
            face_type = self.G.nodes[n].get('face_type', 'Other')
            if face_type == 'Plane':
                node_colors_G.append(self.colors_rgb['geo_plane'])
            elif face_type == 'Cylinder':
                node_colors_G.append(self.colors_rgb['geo_cylinder'])
            else:
                node_colors_G.append(self.colors_rgb['geo_other'])

        node_colors_subG = []
        for n in self.subG.nodes:
            face_type = self.subG.nodes[n].get('face_type', 'Other')
            if face_type == 'Plane':
                node_colors_subG.append(self.colors_rgb['geo_plane'])
            elif face_type == 'Cylinder':
                node_colors_subG.append(self.colors_rgb['geo_cylinder'])
            else:
                node_colors_subG.append(self.colors_rgb['geo_other'])

        # 2.2: Edge colors
        edge_colors_G = []
        for x, y, z in self.G.edges(data=True):
            edge_type = z.get('edge_type', 'other')
            if edge_type == 'convex':
                edge_colors_G.append(self.colors_rgb['edge_convex'])
            elif edge_type == 'concave':
                edge_colors_G.append(self.colors_rgb['edge_concave'])
            elif edge_type == 'tangent':
                edge_colors_G.append(self.colors_rgb['edge_tangent'])
            else:
                edge_colors_G.append((0.5, 0.5, 0.5))  # grey for unknown

        edge_colors_subG = []
        for x, y, z in self.subG.edges(data=True):
            edge_type = z.get('edge_type', 'other')
            if edge_type == 'convex':
                edge_colors_subG.append(self.colors_rgb['edge_convex'])
            elif edge_type == 'concave':
                edge_colors_subG.append(self.colors_rgb['edge_concave'])
            elif edge_type == 'tangent':
                edge_colors_subG.append(self.colors_rgb['edge_tangent'])
            else:
                edge_colors_subG.append((0.5, 0.5, 0.5))  # grey for unknown

        # 3: Final Generation of Graphs
        # 3.1: AAG
        nx.draw_networkx(self.G, pos=nodes_positions, with_labels=True,
            node_color=node_colors_G, edge_color=edge_colors_G, ax=ax1)

        # 3.2: Subgraphs
        nx.draw_networkx(self.subG, pos=nodes_positions, with_labels=True,
            node_color=node_colors_subG, edge_color=edge_colors_subG, ax=ax2)

        '''#4: Features (borders of subgraphs)'''

        # 5: LEGEND
        legend_elements = [
            # Geometry
            Patch(facecolor=self.colors_rgb['geo_plane'], edgecolor='k', label='Plane'),
            Patch(facecolor=self.colors_rgb['geo_cylinder'], edgecolor='k', label='Cylinder'),
            Patch(facecolor=self.colors_rgb['geo_other'], edgecolor='k', label='Other'),
            # Edge types
            Line2D([0], [0], color=self.colors_rgb['edge_convex'], lw=2, label='Convex Edge'),
            Line2D([0], [0], color=self.colors_rgb['edge_concave'], lw=2, label='Concave Edge'),
            Line2D([0], [0], color=self.colors_rgb['edge_tangent'], lw=2, label='Tangent Edge'),
            # Features
            Patch(facecolor='none', edgecolor=self.colors_rgb['feat_hole_through'], linewidth=3, label='Through Hole'),
            Patch(facecolor='none', edgecolor=self.colors_rgb['feat_hole_blind'], linewidth=3, label='Blind Hole'),
            Patch(facecolor='none', edgecolor=self.colors_rgb['feat_pocket_through'], linewidth=3, label='Through Pocket'),
            Patch(facecolor='none', edgecolor=self.colors_rgb['feat_pocket_blind'], linewidth=3, label='Blind Pocket')
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=8, frameon=True)

        for ax in (ax1, ax2):
            ax.axis('off')
        plt.tight_layout()
        plt.show()


