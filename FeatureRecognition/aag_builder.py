import networkx as nx
from typing import List, Dict, Any, Set, Tuple


class AAGBuilder:
    """faces as nodes and adjacency as edges(concave/convex/tangent)"""

    def __init__(self, face_data_list: List[Dict[str, Any]]):
        """face analysis data."""
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

    def find_subgraphs_by_pattern(self, face_types: List[str],
                                  min_connections: int = 1) -> List[Set[int]]:
        """
        Find subgraphs matching a specific pattern of face types.

        Args:
            face_types: List of face types to search for (e.g., ['Cylinder', 'Plane'])
            min_connections: Minimum number of connections required

        Returns:
            List of node sets representing matching subgraphs
        """
        subgraphs = []
        visited = set()

        for node in self.graph.nodes():
            if node in visited:
                continue

            node_type = self.graph.nodes[node]["face_type"]
            if node_type in face_types:
                # Find connected component
                component = self._explore_component(node, face_types, visited)
                if len(component) >= min_connections:
                    subgraphs.append(component)

        return subgraphs

    def _explore_component(self, start_node: int, allowed_types: List[str],
                           visited: Set[int]) -> Set[int]:
        """Explore connected component starting from a node."""
        component = set()
        stack = [start_node]

        while stack:
            node = stack.pop()
            if node in visited:
                continue

            visited.add(node)
            node_type = self.graph.nodes[node]["face_type"]

            if node_type in allowed_types:
                component.add(node)
                # Add neighbors to explore
                for neighbor in self.graph.neighbors(node):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return component

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
