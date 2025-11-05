import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core import TopoDS
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.BRepOffset import BRepOffset_Analyse
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape


def load_step_file(filepath):
    """Load and parse a STEP file, returning the shape object."""
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}")
        return None

    reader = STEPControl_Reader()
    status = reader.ReadFile(filepath)

    if status == IFSelect_RetDone:
        print("STEP file read successfully!")
        reader.TransferRoots()
        shape = reader.OneShape()
        return shape
    else:
        print("Error: Could not read STEP file.")
        return None


def get_face_geometry_type(face: TopoDS_Face):
    """Determine the geometric type of a face (Plane, Cylinder, etc.)."""
    surf_adaptor = BRepAdaptor_Surface(face, True)
    face_type = surf_adaptor.GetType()

    if face_type == GeomAbs_Cylinder:
        return "Cylinder", surf_adaptor.Cylinder()
    elif face_type == GeomAbs_Plane:
        return "Plane", surf_adaptor.Plane()
    else:
        return "Other", None


def get_adjacent_faces(target_face, all_faces_list):
    """Find all faces adjacent to a target face by shared edges."""
    adjacent_faces = []
    target_edges = set()

    edge_explorer = TopExp_Explorer(target_face, TopAbs_EDGE)
    while edge_explorer.More():
        target_edges.add(edge_explorer.Current())
        edge_explorer.Next()

    for face in all_faces_list:
        if face.IsSame(target_face):
            continue

        other_edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while other_edge_explorer.More():
            other_edge = other_edge_explorer.Current()

            for target_edge in target_edges:
                if target_edge.IsSame(other_edge):
                    adjacent_faces.append(face)
                    break

            if face in adjacent_faces:
                break

            other_edge_explorer.Next()

    return adjacent_faces


def classify_edge_type(face1, face2, shared_edge, analyser):
    """Classify edge as Convex, Concave, or Tangent between two faces."""
    convex_edges = TopTools_ListOfShape()
    concave_edges = TopTools_ListOfShape()
    tangent_edges = TopTools_ListOfShape()

    analyser.Edges(face1, 0, concave_edges)
    analyser.Edges(face1, 1, convex_edges)
    analyser.Edges(face1, 2, tangent_edges)

    def occ_list_to_python(occ_list):
        result = []
        iterator = TopTools_ListIteratorOfListOfShape(occ_list)
        while iterator.More():
            result.append(iterator.Value())
            iterator.Next()
        return result

    for edge in occ_list_to_python(convex_edges):
        if TopoDS.Edge(edge).IsSame(shared_edge):
            return "Convex"

    for edge in occ_list_to_python(concave_edges):
        if TopoDS.Edge(edge).IsSame(shared_edge):
            return "Concave"

    for edge in occ_list_to_python(tangent_edges):
        if TopoDS.Edge(edge).IsSame(shared_edge):
            return "Tangent"

    return "Unknown"


def analyze_shape(my_shape):
    """
    Complete analysis of a shape: extract faces, determine types,
    find adjacencies, and classify edges.

    Returns: tuple of (all_faces, face_data_list, analyser)
    """
    analyser = BRepOffset_Analyse(my_shape, 0.01)

    # Extract all faces
    all_faces = []
    face_explorer = TopExp_Explorer(my_shape, TopAbs_FACE)
    while face_explorer.More():
        face = TopoDS.Face(face_explorer.Current())
        all_faces.append(face)
        face_explorer.Next()

    face_to_index_map = {face: i for i, face in enumerate(all_faces)}
    face_data_list = []

    # First pass: geometry types
    for i, face in enumerate(all_faces):
        face_type, geometry = get_face_geometry_type(face)
        face_data_list.append({
            "index": i,
            "face": face,
            "type": face_type,
            "geom": geometry,
            "adjacent_indices": []
        })

    # Second pass: adjacency
    for face_data in face_data_list:
        adjacent_faces = get_adjacent_faces(face_data["face"], all_faces)
        adj_indices = [face_to_index_map[adj_f] for adj_f in adjacent_faces]
        face_data["adjacent_indices"] = adj_indices

    # Third pass: edge classification
    for face_data in face_data_list:
        face_data["convex_adjacent"] = []
        face_data["concave_adjacent"] = []
        face_data["tangent_adjacent"] = []

        face_edges = []
        edge_explorer = TopExp_Explorer(face_data["face"], TopAbs_EDGE)
        while edge_explorer.More():
            face_edges.append(TopoDS.Edge(edge_explorer.Current()))
            edge_explorer.Next()

        classified_faces = set()

        for adj_index in face_data["adjacent_indices"]:
            if adj_index in classified_faces:
                continue

            adj_face = face_data_list[adj_index]["face"]
            adj_edge_explorer = TopExp_Explorer(adj_face, TopAbs_EDGE)
            shared_edge_found = False

            while adj_edge_explorer.More() and not shared_edge_found:
                adj_edge = TopoDS.Edge(adj_edge_explorer.Current())

                for face_edge in face_edges:
                    if face_edge.IsSame(adj_edge):
                        edge_type = classify_edge_type(face_data["face"], adj_face, face_edge, analyser)

                        if edge_type == "Convex":
                            face_data["convex_adjacent"].append(adj_index)
                        elif edge_type == "Concave":
                            face_data["concave_adjacent"].append(adj_index)
                        elif edge_type == "Tangent":
                            face_data["tangent_adjacent"].append(adj_index)

                        classified_faces.add(adj_index)
                        shared_edge_found = True
                        break

                adj_edge_explorer.Next()

    return all_faces, face_data_list, analyser


def print_face_analysis_table(all_faces, face_data_list):
    """Print a formatted table of face analysis results."""
    print("\n--- Model Face Analysis Table ---")
    print(f"Total faces found: {len(all_faces)}")
    print("-------------------------------------------------------------------")
    print(f"{'Face #':<6} | {'Type':<10} | {'Adjacent':<15} | {'Convex':<15} | {'Concave':<15} | {'Tangent'}")
    print("-" * 95)

    for face_data in face_data_list:
        print(f"{face_data['index']:<6} | "
              f"{face_data['type']:<10} | "
              f"{str(face_data['adjacent_indices']):<15} | "
              f"{str(face_data['convex_adjacent']):<15} | "
              f"{str(face_data['concave_adjacent']):<15} | "
              f"{face_data['tangent_adjacent']}")

    print("-------------------------------------------------------------------\n")
