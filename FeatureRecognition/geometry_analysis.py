import os
import numpy as np
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core import TopoDS
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.BRepOffset import BRepOffset_Analyse
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRep import BRep_Tool
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder,
    GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_Hyperbola,
    GeomAbs_Parabola, GeomAbs_BezierCurve, GeomAbs_BSplineCurve,
    GeomAbs_OffsetCurve, GeomAbs_OtherCurve
)



def load_step_file(step_file):
    if not os.path.exists(step_file):
        print("ERROR: NO STEP FILE")
        return None

    reader_occ = STEPControl_Reader() #translate step file info to smth readable by OCC

    if reader_occ.ReadFile(step_file) == IFSelect_RetDone: #reads file --> success or failure
        print("STEP file read successfully!")
        reader_occ.TransferRoots() #step data --> occ internal representation
        shape = reader_occ.OneShape() #shape with all geometry
        return shape
    else:
        print("ERROR: Could not read STEP file.")
        return None

def get_stock_box (shape):
    stock_box = shape.BoundingBox() #CHECK THIS!!!!!!!!!!!!!!!!



def get_face_geometry_type(face):
    adaptor = BRepAdaptor_Surface(face, True) #adapts a face so it can be treated as a surface
    face_type = adaptor.GetType()
    if face_type == GeomAbs_Cylinder:
        return "Cylinder", adaptor.Cylinder()
    elif face_type == GeomAbs_Plane:
        return "Plane", adaptor.Plane()
    else:
        return "Other", None


def get_adjacent_faces(shape, target_face):
    adjacent_faces = set()
    t = TopologyExplorer(shape)
    #get all edges -> get all faces from edge -> adjacent faces
    edges = t.edges_from_face(target_face)

    for edge in edges:
        faces = t.faces_from_edge(edge)
        for face in faces:
            if not face.IsSame(target_face): #the target face is not adjacent to itself
                adjacent_faces.add(face)

    return list(adjacent_faces)




def classify_edge_type(face1, face2, shared_edge, analyser):
    # Prepare occ lists for each type
    convex_edges = TopTools_ListOfShape()
    concave_edges = TopTools_ListOfShape()
    tangent_edges = TopTools_ListOfShape()
    analyser.Edges(face1, 0, concave_edges)
    analyser.Edges(face1, 1, convex_edges)
    analyser.Edges(face1, 2, tangent_edges)
    #convert to python
    def occ_list_to_python(occ_list):
        result = []
        iterator = TopTools_ListIteratorOfListOfShape(occ_list)
        while iterator.More():
            result.append(iterator.Value())
            iterator.Next()
        return result
    #Find where the shared edge fits
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

def get_edge_info(edge):
    adaptor = BRepAdaptor_Curve(edge)
    edge_geom = adaptor.GetType()

    first_param = adaptor.FirstParameter()
    last_param = adaptor.LastParameter()

    is_straight = edge_geom == GeomAbs_Line
    num_points = 50 if not is_straight else 2
    params = np.linspace(first_param, last_param, num_points)

    points = []
    for param in params:
        pnt = adaptor.Value(param)
        points.append([pnt.X(), pnt.Y(), pnt.Z()])

    points = np.array(points)

    edge_length = 0
    for j in range(len(points) - 1):
        edge_length += np.linalg.norm(points[j + 1] - points[j])

    return edge_geom, edge_length



def analyze_shape(my_shape):
    analyser = BRepOffset_Analyse(my_shape, 0.01) #make sre it considers right normals
    t = TopologyExplorer(my_shape)

    all_faces = []
    face_explorer = TopExp_Explorer(my_shape, TopAbs_FACE)
    while face_explorer.More():
        all_faces.append(TopoDS.Face(face_explorer.Current()))
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
            "adjacent_indices": [],
            "convex_adjacent" : [],
            "concave_adjacent" : [],
            "tangent_adjacent" : []
        })


    # Second pass: adjacency
    for face_data in face_data_list:
        adjacent_faces = get_adjacent_faces(my_shape, face_data["face"])
        adj_indices = [face_to_index_map[adj_f] for adj_f in adjacent_faces] #get the indices for the adj faces
        face_data["adjacent_indices"] = adj_indices

    # Third pass: Edge extraction + deduplication

    all_edges_raw = []  # store raw edges as given by OCC
    edge_explorer = TopExp_Explorer(my_shape, TopAbs_EDGE)
    while edge_explorer.More():
        all_edges_raw.append(TopoDS.Edge(edge_explorer.Current()))
        edge_explorer.Next()

    print("\n--- Duplicate Edge Debug Check ---")
    print(f"Total edges detected (raw OCC): {len(all_edges_raw)}")

    # --- REMOVE DUPLICATE EDGES
    unique_edges = []  # will hold only topologically unique edges
    for e in all_edges_raw:
        is_dup = False
        for u in unique_edges:
            if e.IsSame(u):
                is_dup = True
                break
        if not is_dup:
            unique_edges.append(e)

    print(f"Unique edges (after IsSame filtering): {len(unique_edges)}")
    print("------------------------------------------------------\n")

    # Use deduplicated edges from now on
    all_edges = unique_edges  # replace raw list with deduped list

    # Build edge index map for unique edges
    edge_to_index_map = {edge: i for i, edge in enumerate(all_edges)}
    edge_data_list = []

    for i, edge in enumerate(all_edges):
        edge_geom, edge_length = get_edge_info(edge)
        edge_data_list.append({
            "index": i,
            "edge": edge,
            "edge_geom": edge_geom,
            "edge_length": edge_length,
            "classification": []
        })


    # Classify edges per face adjacency
    for face_data in face_data_list:
        face = face_data['face']
        edges_of_face = t.edges_from_face(face)

        for edge in edges_of_face:
            # Match the face-local edge handle to our unique edge list using IsSame()
            matched_index = None  # index in all_edges corresponding to this edge
            for unique_edge in all_edges:
                if edge.IsSame(unique_edge):
                    matched_index = edge_to_index_map[unique_edge]
                    break

            if matched_index is None:
                # skip if no matching unique edge found
                continue

            edge_data = edge_data_list[matched_index]

            adjacent_faces = [f for f in t.faces_from_edge(edge) if not f.IsSame(face)]
            for adj_face in adjacent_faces:
                edge_type = classify_edge_type(face, adj_face, edge, analyser)
                edge_data['classification'].append(edge_type)

                adj_index = face_to_index_map[adj_face]
                if edge_type == "Convex":
                    face_data['convex_adjacent'].append(adj_index)
                elif edge_type == "Concave":
                    face_data['concave_adjacent'].append(adj_index)
                elif edge_type == "Tangent":
                    face_data['tangent_adjacent'].append(adj_index)

    return all_faces, face_data_list, analyser, all_edges, edge_data_list


def print_face_analysis_table(all_faces, face_data_list):
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

def print_edge_analysis_table(all_edges, edge_data_list):
    print("\n--- Model Edge Analysis Table ---")
    print(f"Total edges found: {len(all_edges)}")
    print("-------------------------------------------------------------------")
    print(f"{'Edge #':<6} | {'Geometry':<10} | {'Length':<15} | {'Type':<15} ")
    print("-" * 95)

    for edge_data in edge_data_list:
        print(f"{edge_data['index']:<6} | "
              f"{edge_data['edge_geom']:<10} | "
              f"{str(edge_data['edge_length']):<15} | "
              f"{str(edge_data['classification']):<15}")

    print("-------------------------------------------------------------------\n")


if __name__ == "__main__":

    step_path = r"C:\Users\Maria Salgado\PycharmProjects\Thesis\FeatureRecognition\STEPFiles\example_thoughhole.stp"
    # <---- CHANGE THIS

    shape = load_step_file(step_path)
    if shape is None:
        exit(1)

    all_faces, face_data_list, analyser, all_edges, edge_data_list = analyze_shape(shape)

    # -----------------------------------------------------------
    # Verify that edges are truly unique
    # -----------------------------------------------------------
    print("\n--- Verification: Unique Edge Check in Main ---")
    print(f"Edges returned by analyze_shape(): {len(all_edges)}")

    dup_count = 0
    for i in range(len(all_edges)):
        for j in range(i + 1, len(all_edges)):
            if all_edges[i].IsSame(all_edges[j]):
                dup_count += 1

    if dup_count == 0:
        print("✓ No duplicate edges remain. Deduplication successful!")
    else:
        print(f"✗ WARNING: Found {dup_count} duplicates after analysis.")
    print("-----------------------------------------------------------\n")

    print_face_analysis_table(all_faces, face_data_list)
    print_edge_analysis_table(all_edges, edge_data_list)

