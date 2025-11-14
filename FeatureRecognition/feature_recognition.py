from typing import List, Dict, Any, Tuple
from OCC.Core.TopoDS import TopoDS_Face
from aag_builder import AAGBuilder


class FeatureRecognizer:
    def __init__(self, face_data_list: List[Dict[str, Any]], aag: AAGBuilder):
        """Initialize feature recognizer with face data and AAG."""
        self.face_data_list = face_data_list
        self.aag = aag
        self.recognized_features = []

    def recognize_all_features(self) -> List[Tuple[str, List[TopoDS_Face]]]:
        print("\n--- Starting Feature Recognition ---")

        self.recognized_features = []
        processed_faces = set()

        # Generate subgraph and components
        subgraph = self.aag.get_subgraph_without_convex_edges()
        components = self.aag.get_connected_components(subgraph)
        graph = self.aag.get_graph()

        # Pass components to different feature detectors
        self.recognize_holes(components, graph, processed_faces)
        # self.recognize_pockets(components, graph, processed_faces)
        # self.recognize_slots(components, graph, processed_faces)

        print(f"\n--- Recognized {len(self.recognized_features)} features ---")
        return self.recognized_features

    def recognize_holes(self, components, graph, processed_faces):
        #Through Hole: 1 node (isolated) + cylindrical
        #Blind Hole: 2 nodes, 1 concave edge (cylindrical + planar)

        for component in components:
            if any(idx in processed_faces for idx in component):
                continue

            component_list = list(component)
            face_types = [graph.nodes[idx]['face_type'] for idx in component_list]
            face_objects = [graph.nodes[idx]['face_object'] for idx in component_list]

            # Count concave edges within component
            concave_edges = 0
            for node in component_list:
                concave_neighbors = self.aag.get_concave_neighbors(node)
                for neighbor in concave_neighbors:
                    if neighbor in component and neighbor > node:
                        concave_edges += 1

            # Through Hole
            if len(component) == 1 and face_types[0] == "Cylinder" and concave_edges == 0:
                print(f">>> Recognized 'Through Hole' at face {component_list[0]}")
                self.recognized_features.append(("Through Hole", face_objects))
                processed_faces.update(component)

            # Blind Hole
            elif len(component) == 2 and concave_edges == 1:
                if face_types.count("Cylinder") == 1 and face_types.count("Plane") == 1:
                    print(f">>> Recognized 'Blind Hole' at faces {component_list}")
                    self.recognized_features.append(("Blind Hole", face_objects))
                    processed_faces.update(component)

    def recognize_pockets(self, processed_faces: set):
        """Detect pocket features using AAG patterns."""
        # Implement pocket recognition logic
        # Example: Look for planar faces with concave edges forming a depression
        pass

    def recognize_slots(self, processed_faces: set):
        """Detect slot features."""
        # Implement slot recognition logic
        pass

    def get_feature_statistics(self) -> Dict[str, int]:
        """Return count of each feature type recognized."""
        stats = {}
        for feature_name, _ in self.recognized_features:
            stats[feature_name] = stats.get(feature_name, 0) + 1
        return stats

    def visualize_features(self, display):
        from OCC.Display.OCCViewer import rgb_color

        # Define feature-specific colors
        feature_colors = {
            "Through Hole": rgb_color(0, 0, 0.45),  # Dark Blue
            "Blind Hole": rgb_color(0, 0.3, 1),  # Light Blue
            "Pocket": rgb_color(0.5, 0, 0.5),  # Purple
            "Slot": rgb_color(1, 0.65, 0),  # Orange
        }

        # Default color for unknown features
        default_color = rgb_color(0.7, 0.7, 0.7)  # Light Gray

        for i, (feature_name, faces) in enumerate(self.recognized_features):
            print(f"Feature {i + 1}: {feature_name} (composed of {len(faces)} faces)")

            color = feature_colors.get(feature_name, default_color)
            for face in faces:
                display.DisplayShape(face, update=False, color=color)

        display.FitAll()
        print("Legend: Dark Blue=Through Hole, Light Blue=Blind Hole")

    def visualize_edge_types(self, display):
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core import TopoDS
        from OCC.Display.OCCViewer import rgb_color

        print("\n--- Visualizing Edge Types ---")

        # Create a mapping of edges to their classifications
        edge_classifications = {}

        for face_data in self.face_data_list:
            face = face_data["face"]
            face_edges = []
            edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)

            while edge_explorer.More():
                edge = TopoDS.Edge(edge_explorer.Current())
                face_edges.append(edge)
                edge_explorer.Next()

            # Check each adjacent face for shared edges
            for adj_idx in face_data["convex_adjacent"]:
                adj_face = self.face_data_list[adj_idx]["face"]
                shared_edge = self._find_shared_edge(face_edges, adj_face)
                if shared_edge:
                    edge_classifications[shared_edge] = "Convex"

            for adj_idx in face_data["concave_adjacent"]:
                adj_face = self.face_data_list[adj_idx]["face"]
                shared_edge = self._find_shared_edge(face_edges, adj_face)
                if shared_edge:
                    edge_classifications[shared_edge] = "Concave"

            for adj_idx in face_data["tangent_adjacent"]:
                adj_face = self.face_data_list[adj_idx]["face"]
                shared_edge = self._find_shared_edge(face_edges, adj_face)
                if shared_edge:
                    edge_classifications[shared_edge] = "Tangent"

        # Display edges with appropriate colors
        color_map = {
            "Convex": rgb_color(0, 0, 1),  # Blue
            "Concave": rgb_color(1, 0, 0),  # Red
            "Tangent": rgb_color(0, 1, 0)  # Green
        }

        for edge, edge_type in edge_classifications.items():
            color = color_map.get(edge_type, rgb_color(0.5, 0.5, 0.5))  # Gray
            display.DisplayShape(edge, update=False, color=color)

        print(f"Displayed {len(edge_classifications)} classified edges")
        print("Legend: RED=Concave, BLUE=Convex, GREEN=Tangent")
        display.FitAll()

    def _find_shared_edge(self, face_edges, adj_face):
        """Find the shared edge between a face's edges and an adjacent face."""
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core import TopoDS

        adj_edge_explorer = TopExp_Explorer(adj_face, TopAbs_EDGE)

        while adj_edge_explorer.More():
            adj_edge = TopoDS.Edge(adj_edge_explorer.Current())

            for face_edge in face_edges:
                if face_edge.IsSame(adj_edge):
                    return face_edge

            adj_edge_explorer.Next()

        return None
