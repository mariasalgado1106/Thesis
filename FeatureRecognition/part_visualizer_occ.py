from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import topods  # Import topods module
from OCC.Display.OCCViewer import rgb_color


class PartVisualizer:
    def __init__(self, builder, recognizer):
        self.builder = builder
        self.recognizer = recognizer

    def display_base_shape(self, display, transparency=0.8):
        base_color = self.builder.colors_rgb.get('geo_plane', (0.9, 0.9, 0.9))
        occ_color = rgb_color(*base_color)
        display.DisplayShape(self.builder.shape, update=False,
                             color=occ_color, transparency=transparency)

    def find_shared_edge(self, face_edges, adj_face):
        edge_explorer_adj = TopExp_Explorer(adj_face, TopAbs_EDGE)
        while edge_explorer_adj.More():
            adj_edge = topods.Edge(edge_explorer_adj.Current())
            for face_edge in face_edges:
                if face_edge.IsSame(adj_edge):
                    return adj_edge
            edge_explorer_adj.Next()
        return None

    def visualize_edges(self, display):
        print("Visualizing Edges")
        processed_edges = set()

        for face_data in self.builder.face_data_list:
            current_face = face_data["face"]
            current_idx = face_data["index"]

            # edges from current face
            face_edges = []
            edge_explorer = TopExp_Explorer(current_face, TopAbs_EDGE)
            while edge_explorer.More():
                edge = topods.Edge(edge_explorer.Current())
                face_edges.append(edge)
                edge_explorer.Next()

            def draw_edges(adj_indices, color):
                for adj_idx in adj_indices:
                    pair_faces = tuple(sorted((current_idx, adj_idx)))
                    if pair_faces in processed_edges:
                        continue

                    adj_face = self.builder.face_data_list[adj_idx]["face"]
                    shared_edge = self.find_shared_edge(face_edges, adj_face)

                    if shared_edge:
                        edge_color = self.builder.colors_rgb.get(color)
                        occ_color = rgb_color(*edge_color)
                        display.DisplayShape(shared_edge, update=False, color=occ_color)
                        processed_edges.add(pair_faces)

            draw_edges(face_data["convex_adjacent"], "edge_convex")
            draw_edges(face_data["concave_adjacent"], "edge_concave")
            draw_edges(face_data["tangent_adjacent"], "edge_tangent")



        print(f"\nTotal edges displayed: {len(processed_edges)}")
        display.FitAll()