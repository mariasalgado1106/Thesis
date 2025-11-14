import networkx as nx
from typing import List, Dict, Any, Set, Tuple


class AAGBuilder:
    #faces=nodes and adjacency=edges

    def __init__(self, face_data_list: List[Dict[str, Any]]):
        self.face_data_list = face_data_list
        self.graph = nx.Graph()
        self._build_graph()

    def _build_graph(self):
        """Construct the AAG from face data."""
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
        """Return the constructed AAG."""
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



