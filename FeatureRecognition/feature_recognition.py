from typing import List, Dict, Any, Tuple
from OCC.Core.TopoDS import TopoDS_Face
from aag_builder import AAGBuilder


class FeatureRecognizer:
    """
    Feature recognition engine using AAG and rule-based patterns.
    """

    def __init__(self, face_data_list: List[Dict[str, Any]], aag: AAGBuilder):
        """Initialize feature recognizer with face data and AAG."""
        self.face_data_list = face_data_list
        self.aag = aag
        self.recognized_features = []

    def recognize_all_features(self) -> List[Tuple[str, List[TopoDS_Face]]]:
        """Run all feature recognition rules."""
        print("\n--- Starting Feature Recognition ---")

        self.recognized_features = []
        processed_faces = set()

        # Run different feature detectors
        self._recognize_holes(processed_faces)
        # Add more feature recognition methods here:
        # self._recognize_pockets(processed_faces)
        # self._recognize_slots(processed_faces)
        # self._recognize_steps(processed_faces)

        print(f"\n--- Recognized {len(self.recognized_features)} features ---")
        return self.recognized_features

    def _recognize_holes(self, processed_faces: set):
        """Detect through holes and blind holes."""
        for face_data in self.face_data_list:
            face = face_data["face"]
            if face in processed_faces:
                continue

            # Rule: Cylinder with at least 2 planar neighbors
            if face_data["type"] == "Cylinder":
                print(f"Found potential cylinder at face {face_data['index']}...")

                adj_face_indices = face_data["adjacent_indices"]
                planar_neighbors = 0

                for adj_index in adj_face_indices:
                    adj_face_data = self.face_data_list[adj_index]
                    if adj_face_data["type"] == "Plane":
                        planar_neighbors += 1

                if planar_neighbors >= 2:
                    print(f">>> Recognized 'Hole' feature at face {face_data['index']}!")

                    feature_faces = [face]

                    # Add adjacent planar faces
                    for adj_index in adj_face_indices:
                        if self.face_data_list[adj_index]["type"] == "Plane":
                            feature_faces.append(self.face_data_list[adj_index]["face"])

                    self.recognized_features.append(("Hole", feature_faces))

                    for f in feature_faces:
                        processed_faces.add(f)

    def _recognize_pockets(self, processed_faces: set):
        """Detect pocket features using AAG patterns."""
        # Implement pocket recognition logic
        # Example: Look for planar faces with concave edges forming a depression
        pass

    def _recognize_slots(self, processed_faces: set):
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
        """
        Visualize recognized features with different colors.

        Args:
            display: OCC display object for rendering
        """
        colors = ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "CYAN", "MAGENTA", "BLACK"]

        for i, (feature_name, faces) in enumerate(self.recognized_features):
            print(f"Feature {i + 1}: {feature_name} (composed of {len(faces)} faces)")

            color = colors[i % len(colors)]
            for face in faces:
                display.DisplayShape(face, update=False, color=color)

        display.FitAll()

    def visualize_edge_types(self, display):
        """
        Visualize edges colored by their type (concave/convex/tangent).
        This helps understand the geometric relationships.
        """
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopAbs import TopAbs_EDGE
        from OCC.Core import TopoDS

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
            "Convex": "BLUE",
            "Concave": "RED",
            "Tangent": "GREEN"
        }

        for edge, edge_type in edge_classifications.items():
            color = color_map.get(edge_type, "GRAY")
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
