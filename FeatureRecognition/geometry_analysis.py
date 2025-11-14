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
from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.BRep import BRep_Tool


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


def analyze_shape(my_shape):
    analyser = BRepOffset_Analyse(my_shape, 0.01)
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

    # Third pass: edge classification
    for face_data in face_data_list:
        face = face_data['face']
        edges = t.edges_from_face(face)
        for edge in edges:
            adjacent_faces = [f for f in t.faces_from_edge(edge) if not f.IsSame(face)] #excludes actual afce
            for adj_face in adjacent_faces:
                edge_type = classify_edge_type(face, adj_face, edge, analyser)
                adj_index = face_to_index_map[adj_face]
                if edge_type == "Convex":
                    face_data['convex_adjacent'].append(adj_index)
                elif edge_type == "Concave":
                    face_data['concave_adjacent'].append(adj_index)
                elif edge_type == "Tangent":
                    face_data['tangent_adjacent'].append(adj_index)

    return all_faces, face_data_list, analyser


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
