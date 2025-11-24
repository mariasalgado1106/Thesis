import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from geometry_analysis import (load_step_file, analyze_shape)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

class AAGBuilder_3D:
    def __init__(self, my_shape):
        self.my_shape = my_shape

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
                       adjacent_faces=face_data["adjacent_indices"])

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
        def filter_edge(n1, n2): #n1 and n2 are the 2 nodes
            return self.G[n1][n2].get("edge_type") != "convex" #if convex it returns false and removes
        self.subG = nx.subgraph_view(self.G, filter_edge=filter_edge)

        return self.subG

    # 2. ANALYSE SUBGRAPH FOR FR (connected faces)

    def analyse_subgraphs (self):
        if self.subG is None:
            self.build_aag_subgraph()
        subgraphs = list(nx.connected_components(self.subG)) #the subgraphs/components
        self.subgraphs_info = []

        for i, nodeset in enumerate(subgraphs): #i=nr of the component/subgraph
            sg = self.subG.subgraph(nodeset)
            nodes = list(sg.nodes())
            n_faces = len(nodes)
            face_types = [self.face_data_list[node]['type'] for node in nodes]
            n_concave = sum(1 for _, _, type in sg.edges(data=True) if type.get('edge_type') == 'concave')
            print(f"Subgraph {i}: faces={n_faces}, concave_edges={n_concave}")

            self.subgraphs_info.append({
                'subgraph_idx': i,
                'nodes': nodes,
                'n_faces': n_faces,
                'n_concave': n_concave,
                'face_types': face_types
            })
        return self.subgraphs_info


    # 3. VISUALIZE GRAPHS
    def visualize_aag (self):
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
        ax2.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

        for ax in (ax1, ax2):
            ax.axis('off')
        plt.tight_layout()
        plt.show()


