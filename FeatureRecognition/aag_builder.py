import networkx as nx
from typing import List, Dict, Any, Set, Tuple
from geometry_analysis import (load_step_file, analyze_shape)
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


# 1. BUILD GRAPH AND SUBGRAPH
def build_aag_graph(my_shape): #graph with all edge types
    G = nx.Graph()
    all_faces, face_data_list, _ = analyze_shape(my_shape)

    for face_data in face_data_list:
        i = face_data["index"]
        G.add_node(i, face_type=face_data["type"], geometry=face_data["geom"],
                   adjacent_faces=face_data["adjacent_indices"])

    # Add ALL edge types
    for face_data in face_data_list:
        current_face = face_data["index"]

        for adj_idx in face_data["convex_adjacent"]:
            if not G.has_edge(current_face, adj_idx):
                G.add_edge(current_face, adj_idx, edge_type="convex")

        for adj_idx in face_data["concave_adjacent"]:
            if not G.has_edge(current_face, adj_idx):
                G.add_edge(current_face, adj_idx, edge_type="concave")

        for adj_idx in face_data["tangent_adjacent"]:
            if not G.has_edge(current_face, adj_idx):
                G.add_edge(current_face, adj_idx, edge_type="tangent")

    print(f"TRIAL Total nodes: {G.number_of_nodes()}")
    print(f"TRIAL Total edges (complete graph): {G.number_of_edges()}")

    convex_count = sum(1 for x, y, z in G.edges(data=True) if z.get('edge_type') == 'convex')
    print(f"TRIAL Total convex edges: {convex_count}")

    concave_count = sum(1 for x, y, z in G.edges(data=True) if z.get('edge_type') == 'concave')
    print(f"TRIAL Total concave edges: {concave_count}")

    tangent_count = sum(1 for x, y, z in G.edges(data=True) if z.get('edge_type') == 'tangent')
    print(f"TRIAL Total tangent edges: {tangent_count}")

    return G


def build_aag_subgraph (my_shape):
    G = build_aag_graph(my_shape)#subgraph without convex
    def filter_edge(n1, n2): #n1 and n2 are the 2 nodes
        return G[n1][n2].get("edge_type") != "convex" #if convex it returns false and removes
    subG = nx.subgraph_view(G, filter_edge=filter_edge)

    return subG

# 2. ANALYSE SUBGRAPH FOR FR (connected faces)

def analyse_subgraphs (subG, face_data_list):
    subgraphs = list(nx.connected_components(subG)) #the subgraphs/components
    subgraphs_info = []

    for i, nodeset in enumerate(subgraphs): #i=nr of the component/subgraph
        sg = subG.subgraph(nodeset)
        nodes = list(sg.nodes())
        n_faces = len(nodes)
        face_types = [face_data_list[node]['type'] for node in nodes]
        n_concave = sum(1 for _, _, type in sg.edges(data=True) if type.get('edge_type') == 'concave')
        print(f"Subgraph {i}: faces={n_faces}, concave_edges={n_concave}")

        subgraphs_info.append({
            'subgraph_idx': idx,
            'nodes': nodes,
            'n_faces': n_faces,
            'n_concave': n_concave,
            'face_types': face_types
        })
    return subgraphs_info


# 3. VISUALIZE GRAPHS
def visualize_aag (my_shape):
    G = build_aag_graph(my_shape)
    subG = build_aag_subgraph(my_shape)

    nodes_positions = nx.spring_layout(G, seed=42) #aesthetic way of representing the nodes

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7)) #side by side plots

    # 1: Define colors
    colors_rgb = {
        # EDGES
        "edge_concave": (1.0, 0.0, 0.0),  # Red
        "edge_convex": (0.0, 0.0, 1.0),  # Blue
        "edge_tangent": (0.0, 0.0, 0.0),  # Black

        # FEATURES (Node Borders)
        "feat_hole_through": (0.000, 0.000, 0.45), #Dark Blue
        "feat_hole_blind": (0, 0.3, 1), #Light Blue
        "feat_pocket_through": (0.000, 0.392, 0.000), #Dark Green
        "feat_pocket_blind": (0.196, 0.804, 0.196), #Light Green
        "feat_slot_through": (1.000, 0.078, 0.576), #Dark Pink
        "feat_slot_blind": (1.000, 0.412, 0.706), #Hot Pink
        "feat_step": (1.000, 0.549, 0.000), #Dark Orange
        "feat_other": (1.000, 0.2, 0.1), #Light Orange

        # GEOMETRY (Node Fills)
        "geo_plane": (0.961, 0.961, 0.961),  # Light Grey
        "geo_cylinder": (1.000, 0.980, 0.804),  # Pale Yellow
        "geo_other": (0.980, 0.941, 0.902),  # Beige
    }

    # 2: Create Graphs
    # 2.1: Colors by geometry
    node_colors_G = []
    for n in G.nodes:
        face_type = G.nodes[n].get('face_type', 'Other')
        if face_type == 'Plane':
            node_colors_G.append(colors_rgb['geo_plane'])
        elif face_type == 'Cylinder':
            node_colors_G.append(colors_rgb['geo_cylinder'])
        else:
            node_colors_G.append(colors_rgb['geo_other'])

    node_colors_subG = []
    for n in subG.nodes:
        face_type = subG.nodes[n].get('face_type', 'Other')
        if face_type == 'Plane':
            node_colors_subG.append(colors_rgb['geo_plane'])
        elif face_type == 'Cylinder':
            node_colors_subG.append(colors_rgb['geo_cylinder'])
        else:
            node_colors_subG.append(colors_rgb['geo_other'])

    # 2.2: Edge colors
    edge_colors_G = []
    for x, y, z in G.edges(data=True):
        edge_type = z.get('edge_type', 'other')
        if edge_type == 'convex':
            edge_colors_G.append(colors_rgb['edge_convex'])
        elif edge_type == 'concave':
            edge_colors_G.append(colors_rgb['edge_concave'])
        elif edge_type == 'tangent':
            edge_colors_G.append(colors_rgb['edge_tangent'])
        else:
            edge_colors_G.append((0.5, 0.5, 0.5))  # grey for unknown

    edge_colors_subG = []
    for x, y, z in subG.edges(data=True):
        edge_type = z.get('edge_type', 'other')
        if edge_type == 'convex':
            edge_colors_subG.append(colors_rgb['edge_convex'])
        elif edge_type == 'concave':
            edge_colors_subG.append(colors_rgb['edge_concave'])
        elif edge_type == 'tangent':
            edge_colors_subG.append(colors_rgb['edge_tangent'])
        else:
            edge_colors_subG.append((0.5, 0.5, 0.5))  # grey for unknown

    # 3: Final Generation of Graphs
    # 3.1: AAG
    nx.draw_networkx(G, pos=nodes_positions, with_labels=True,
        node_color=node_colors_G, edge_color=edge_colors_G, ax=ax1)

    # 3.2: Subgraphs
    nx.draw_networkx(subG, pos=nodes_positions, with_labels=True,
        node_color=node_colors_subG, edge_color=edge_colors_subG, ax=ax2)

    '''#4: Features (borders of subgraphs)'''

    # 5: LEGEND
    legend_elements = [
        # Geometry
        Patch(facecolor=colors_rgb['geo_plane'], edgecolor='k', label='Plane'),
        Patch(facecolor=colors_rgb['geo_cylinder'], edgecolor='k', label='Cylinder'),
        Patch(facecolor=colors_rgb['geo_other'], edgecolor='k', label='Other'),
        # Edge types
        Line2D([0], [0], color=colors_rgb['edge_convex'], lw=2, label='Convex Edge'),
        Line2D([0], [0], color=colors_rgb['edge_concave'], lw=2, label='Concave Edge'),
        Line2D([0], [0], color=colors_rgb['edge_tangent'], lw=2, label='Tangent Edge'),
        # Features
        Patch(facecolor='none', edgecolor=colors_rgb['feat_hole_through'], linewidth=3, label='Through Hole'),
        Patch(facecolor='none', edgecolor=colors_rgb['feat_hole_blind'], linewidth=3, label='Blind Hole'),
        Patch(facecolor='none', edgecolor=colors_rgb['feat_pocket_through'], linewidth=3, label='Through Pocket'),
        Patch(facecolor='none', edgecolor=colors_rgb['feat_pocket_blind'], linewidth=3, label='Blind Pocket')
    ]
    ax2.legend(handles=legend_elements, loc='lower right', fontsize=8, frameon=True)

    for ax in (ax1, ax2):
        ax.axis('off')
    plt.tight_layout()
    plt.show()












































class AAGBuilder:
    #faces=nodes and adjacency=edges
    def __init__(self, face_data_list: List[Dict[str, Any]]):
        self.face_data_list = face_data_list
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        # Add nodes with face attributes
        for face_data in self.face_data_list:
            self.graph.add_node(
                face_data["index"],
                face_type=face_data["type"],
                face_object=face_data["face"],
                geometry=face_data["geom"]
            )

        # Add edges with convexity attributes
        for face_data in self.face_data_list:
            face_idx = face_data["index"]

            # Add convex edges
            for adj_idx in face_data["convex_adjacent"]:
                if not self.graph.has_edge(face_idx, adj_idx):
                    self.graph.add_edge(face_idx, adj_idx, edge_type="Convex")

            # Add concave edges
            for adj_idx in face_data["concave_adjacent"]:
                if not self.graph.has_edge(face_idx, adj_idx):
                    self.graph.add_edge(face_idx, adj_idx, edge_type="Concave")

            # Add tangent edges
            for adj_idx in face_data["tangent_adjacent"]:
                if not self.graph.has_edge(face_idx, adj_idx):
                    self.graph.add_edge(face_idx, adj_idx, edge_type="Tangent")

    def get_graph(self) -> nx.Graph:
        return self.graph

    def get_concave_neighbors(self, node_idx: int) -> List[int]:
        """Get all neighbors connected by concave edges."""
        neighbors = []
        for neighbor in self.graph.neighbors(node_idx):
            if self.graph[node_idx][neighbor].get("edge_type") == "Concave":
                neighbors.append(neighbor)
        return neighbors

    def get_convex_neighbors(self, node_idx: int) -> List[int]:
        """Get all neighbors connected by convex edges."""
        neighbors = []
        for neighbor in self.graph.neighbors(node_idx):
            if self.graph[node_idx][neighbor].get("edge_type") == "Convex":
                neighbors.append(neighbor)
        return neighbors

    def get_subgraph_without_convex_edges(self) -> nx.Graph:
        def filter_edge(n1, n2):
            return self.graph[n1][n2].get("edge_type") != "Convex"

        return nx.subgraph_view(self.graph, filter_edge=filter_edge)

    def get_connected_components(self, subgraph: nx.Graph = None) -> List[Set[int]]:
        graph_to_use = subgraph if subgraph is not None else self.graph
        return [set(component) for component in nx.connected_components(graph_to_use)]

    def print_graph_summary(self):
        """Print summary statistics of the AAG."""
        print("\n--- AAG Summary ---")
        print(f"Total nodes (faces): {self.graph.number_of_nodes()}")
        print(f"Total edges (adjacencies): {self.graph.number_of_edges()}")

        # Count edge types
        convex_count = sum(1 for _, _, data in self.graph.edges(data=True)
                           if data.get("edge_type") == "Convex")
        concave_count = sum(1 for _, _, data in self.graph.edges(data=True)
                            if data.get("edge_type") == "Concave")
        tangent_count = sum(1 for _, _, data in self.graph.edges(data=True)
                            if data.get("edge_type") == "Tangent")

        print(f"Convex edges: {convex_count}")
        print(f"Concave edges: {concave_count}")
        print(f"Tangent edges: {tangent_count}")
        print("-------------------\n")

    def export_graph_data(self) -> Dict[str, Any]:
        """Export graph data for visualization or further processing."""
        nodes_data = []
        for node, data in self.graph.nodes(data=True):
            nodes_data.append({
                "id": node,
                "type": data["face_type"]
            })

        edges_data = []
        for u, v, data in self.graph.edges(data=True):
            edges_data.append({
                "source": u,
                "target": v,
                "edge_type": data.get("edge_type", "Unknown")
            })

        return {
            "nodes": nodes_data,
            "edges": edges_data
        }

    def visualize_graph(self, recognized_features=None, save_path=None):
        import matplotlib.pyplot as plt
        import networkx as nx

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 8))

        # Plot 1: Full AAG with all edges
        ax1.set_title("Full AAG (All Edges)", fontsize=16, fontweight='bold')
        pos = nx.spring_layout(self.graph, k=2, iterations=50, seed=42)

        # Draw nodes colored by face type with gray shades
        face_types = nx.get_node_attributes(self.graph, 'face_type')
        node_colors = []
        for node in self.graph.nodes():
            ftype = face_types.get(node, "Unknown")
            if ftype == "Cylinder":
                node_colors.append((0.7, 0.7, 0.7))  # Light Gray
            elif ftype == "Plane":
                node_colors.append((0.5, 0.5, 0.5))  # Medium Gray
            else:
                node_colors.append((0.3, 0.3, 0.3))  # Dark Gray

        nx.draw_networkx_nodes(self.graph, pos, node_color=node_colors, node_size=500, ax=ax1)
        nx.draw_networkx_labels(self.graph, pos, font_size=8, ax=ax1)

        # Draw edges colored by type
        convex_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                        if d.get('edge_type') == 'Convex']
        concave_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                         if d.get('edge_type') == 'Concave']
        tangent_edges = [(u, v) for u, v, d in self.graph.edges(data=True)
                         if d.get('edge_type') == 'Tangent']

        nx.draw_networkx_edges(self.graph, pos, edgelist=convex_edges,
                               edge_color=(0, 0, 1), width=2, label='Convex', ax=ax1)  # blue
        nx.draw_networkx_edges(self.graph, pos, edgelist=concave_edges,
                               edge_color=(1, 0, 0), width=2, label='Concave', ax=ax1)  # red
        nx.draw_networkx_edges(self.graph, pos, edgelist=tangent_edges,
                               edge_color=(0, 1, 0), width=2, label='Tangent', ax=ax1)  # green

        # Add legend for Plot 1 (combine face types and edge types)
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D

        legend_elements_ax1 = [
            # Face types
            Patch(facecolor=(0.5, 0.5, 0.5), label='Plane'),
            Patch(facecolor=(0.7, 0.7, 0.7), label='Cylinder'),
            # Edge types
            Line2D([0], [0], color=(0, 0, 1), linewidth=2, label='Convex'),
            Line2D([0], [0], color=(1, 0, 0), linewidth=2, label='Concave'),
            Line2D([0], [0], color=(0, 1, 0), linewidth=2, label='Tangent'),
        ]
        ax1.legend(handles=legend_elements_ax1, loc='upper left', fontsize=8)
        ax1.axis('off')

        # Plot 2: Subgraph without convex edges + recognized features
        ax2.set_title("Subgraph (No Convex Edges) - Features Highlighted",
                      fontsize=16, fontweight='bold')

        subgraph = self.get_subgraph_without_convex_edges()
        components = self.get_connected_components(subgraph)

        # Create feature mapping
        feature_map = {}
        if recognized_features:
            for feature_name, _, component_indices in recognized_features:
                for idx in component_indices:
                    feature_map[idx] = feature_name

        # Color nodes by recognized feature
        node_colors_sub = []
        for node in subgraph.nodes():
            feature = feature_map.get(node)
            if feature == "Through Hole":
                node_colors_sub.append((0, 0, 0.45))  # Dark Blue
            elif feature == "Blind Hole":
                node_colors_sub.append((0, 0.3, 1))  # Light Blue
            elif feature == "Through Pocket":
                node_colors_sub.append((0, 0.39, 0))  # Dark Green
            elif feature == "Blind Pocket":
                node_colors_sub.append((0.56, 0.93, 0.56))  # Light Green
            elif feature == "Through Slot":
                node_colors_sub.append((1, 0, 0))  # Red
            elif feature == "Blind Slot":
                node_colors_sub.append((1, 0.75, 0.80))  # Pink
            elif feature == "Through Step":
                node_colors_sub.append((1, 0.65, 0))  # Orange
            elif feature == "Blind Step":
                node_colors_sub.append((1, 1, 0))  # Yellow
            else:
                # Unrecognized - color by face type with gray shades
                ftype = face_types.get(node, "Unknown")
                if ftype == "Cylinder":
                    node_colors_sub.append((0.7, 0.7, 0.7))  # Light Gray
                elif ftype == "Plane":
                    node_colors_sub.append((0.5, 0.5, 0.5))  # Medium Gray
                else:
                    node_colors_sub.append((0.3, 0.3, 0.3))  # Dark Gray

        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors_sub,
                               node_size=500, ax=ax2)
        nx.draw_networkx_labels(subgraph, pos, font_size=8, ax=ax2)

        # Draw only concave and tangent edges
        concave_edges_sub = [(u, v) for u, v, d in subgraph.edges(data=True)
                             if d.get('edge_type') == 'Concave']
        tangent_edges_sub = [(u, v) for u, v, d in subgraph.edges(data=True)
                             if d.get('edge_type') == 'Tangent']

        nx.draw_networkx_edges(subgraph, pos, edgelist=concave_edges_sub,
                               edge_color=(1, 0, 0), width=2, label='Concave', ax=ax2)  # red
        nx.draw_networkx_edges(subgraph, pos, edgelist=tangent_edges_sub,
                               edge_color=(0, 1, 0), width=2, label='Tangent', ax=ax2)  # green

        # Add legend for features
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=(0, 0, 0.45), label='Through Hole'),  # Dark Blue
            Patch(facecolor=(0, 0.3, 1), label='Blind Hole'),  # Light Blue
            Patch(facecolor=(0, 0.39, 0), label='Through Pocket'),  # Dark Green
            Patch(facecolor=(0.56, 0.93, 0.56), label='Blind Pocket'),  # Light Green
            Patch(facecolor=(1, 0, 0), label='Through Slot'),  # Red
            Patch(facecolor=(1, 0.75, 0.80), label='Blind Slot'),  # Pink
            Patch(facecolor=(1, 0.65, 0), label='Through Step'),  # Orange
            Patch(facecolor=(1, 1, 0), label='Blind Step'),  # Yellow
            Patch(facecolor=(0.5, 0.5, 0.5), label='Plane'),  # Medium Gray
            Patch(facecolor=(0.7, 0.7, 0.7), label='Cylinder'),  # Light Gray
        ]
        ax2.legend(handles=legend_elements, loc='upper left', fontsize=8)
        ax2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Graph visualization saved to {save_path}")

        plt.show()

        # Print component statistics
        print(f"\nConnected Components: {len(components)}")
        for i, comp in enumerate(components):
            feature_names = [feature_map.get(idx, "Unrecognized") for idx in comp]
            print(f"  Component {i + 1}: {comp} - {set(feature_names)}")



