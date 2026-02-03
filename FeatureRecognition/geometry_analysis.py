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
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop

from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib

from OCC.Core.gp import gp_Pnt, gp_Vec

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.TopLoc import TopLoc_Location
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


def get_stock_box(shape, tol=1e-6):
    bbox = Bnd_Box()
    bbox.SetGap(tol)  # small tolerance to avoid precision issues
    brepbndlib.Add(shape, bbox)  # Add shape to bounding box
    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    cz = 0.5 * (zmin + zmax)
    stock_box_center = gp_Pnt(cx, cy, cz)
    return xmin, ymin, zmin, xmax, ymax, zmax, stock_box_center


def get_face_geometry(face):
    adaptor = BRepAdaptor_Surface(face, True) #adapts a face so it can be treated as a surface
    face_type = adaptor.GetType()
    if face_type == GeomAbs_Cylinder:
        return "Cylinder", adaptor.Cylinder()
    elif face_type == GeomAbs_Plane:
        return "Plane", adaptor.Plane()
    else:
        return "Other", None


def define_stock_face(face_data):
    # Plane and has NO concave neighbors, it's a Stock Face
    if face_data["type"] != "Plane":
        return "No"

    if len(face_data['concave_adjacent']) == 0:
        return "Yes"
    else:
        return "No"



    '''
    xmin, ymin, zmin, xmax, ymax, zmax, stock_box_center = get_stock_box(shape, 1e-6)
    face_center_coords, _ = get_face_center(face)
    x, y, z = face_center_coords
    
    face_xmin = abs(x - xmin) <= tol
    face_xmax = abs(x - xmax) <= tol
    face_ymin = abs(y - ymin) <= tol
    face_ymax = abs(y - ymax) <= tol
    face_zmin = abs(z - zmin) <= tol
    face_zmax = abs(z - zmax) <= tol

    if face_xmin or face_xmax or face_ymin or face_ymax or face_zmin or face_zmax:
        return "Yes"
    else:
        return "No"
    '''

def normal_vector_face (face, shape):
    _, _, _, _, _, _, stock_box_center = get_stock_box(shape, 1e-6)
    _, face_center = get_face_center(face)

    surf = BRepAdaptor_Surface(face, True)
    umin, umax = surf.FirstUParameter(), surf.LastUParameter()
    vmin, vmax = surf.FirstVParameter(), surf.LastVParameter()
    u = 0.5 * (umin + umax)
    v = 0.5 * (vmin + vmax)

    p = gp_Pnt()
    d1u = gp_Vec()
    d1v = gp_Vec()
    surf.D1(u, v, p, d1u, d1v) #this gives values to p, d1u and d1v

    n = d1u.Crossed(d1v)
    if n.Magnitude() == 0:
        return 0.0, 0.0, 0.0
    n.Normalize()

    # direction vector from face_center to stock center
    vv = gp_Vec(stock_box_center.XYZ() - face_center.XYZ())

    #normal to be consistent toward box center
    if n.Dot(vv) < 0:
        n = gp_Vec(-n.X(), -n.Y(), -n.Z())

    n_coords = ([n.X(), n.Y(), n.Z()])

    def normal_axis_direction(n_coords, tol=1e-3):
        nx, ny, nz = n_coords
        if abs(nx - 1.0) < tol and abs(ny) < tol and abs(nz) < tol:
            return "x"
        elif abs(nx + 1.0) < tol and abs(ny) < tol and abs(nz) < tol:
            return "-x"
        elif abs(ny - 1.0) < tol and abs(nx) < tol and abs(nz) < tol:
            return "y"
        elif abs(ny + 1.0) < tol and abs(nx) < tol and abs(nz) < tol:
            return "-y"
        elif abs(nz - 1.0) < tol and abs(nx) < tol and abs(ny) < tol:
            return "z"
        elif abs(nz + 1.0) < tol and abs(nx) < tol and abs(ny) < tol:
            return "-z"
        else:
            return "No axis"

    n_axis = normal_axis_direction(n_coords, tol=1e-3)
    return n, n_coords, n_axis


def get_face_center (face):
    props = GProp_GProps()
    brepgprop.SurfaceProperties(face, props)
    center = props.CentreOfMass()
    face_center_coords = ([center.X(), center.Y(), center.Z()])
    return face_center_coords, center

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


def triangulate_face(face, linear_deflection=0.1):
    #Triangulate a face and return vertices and triangles

    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, location)

    if triangulation is None:
        return [], []

    transformation = location.Transformation()

    # vertices
    vertices = []
    for i in range(1, triangulation.NbNodes() + 1):
        pnt = triangulation.Node(i)
        pnt.Transform(transformation)
        vertices.append([pnt.X(), pnt.Y(), pnt.Z()])

    # triangles (convert to 0-based indexing)
    triangles = []
    for i in range(1, triangulation.NbTriangles() + 1):
        triangle = triangulation.Triangle(i)
        n1, n2, n3 = triangle.Get()
        triangles.append([n1 - 1, n2 - 1, n3 - 1])

    return vertices, triangles


def triangulate_shape(shape, linear_deflection=0.1):
    #Triangulate the entire shape before extracting face meshes
    #Call this once at the beginning

    mesher = BRepMesh_IncrementalMesh(shape, linear_deflection)
    mesher.Perform()
    if not mesher.IsDone():
        print("Warning: Meshing may be incomplete")


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

    return edge_geom, edge_length, points




def analyze_shape(my_shape):
    # First, triangulate the entire shape
    linear_deflection = 0.1
    triangulate_shape(my_shape, linear_deflection)
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
        face_type, geometry = get_face_geometry(face)
        face_center, _ = get_face_center(face)

        n, n_coords, n_axis = normal_vector_face(face, my_shape)
        vertices, triangles = triangulate_face(face, linear_deflection) #mesh triangulation
        face_data_list.append({
            "index": i,
            "face": face,
            "type": face_type,
            "geom": geometry,
            "face_center": face_center,
            "stock_face": "Pending",
            "adjacent_indices": [],
            "convex_adjacent" : [],
            "concave_adjacent" : [],
            "tangent_adjacent" : [],
            "normal_vector": n,
            "normal_vector_coords": n_coords,
            "normal_vector_axis": n_axis,
            "mesh_vertices": vertices,
            "mesh_triangles": triangles
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
        edge_geom, edge_length, points = get_edge_info(edge)
        edge_data_list.append({
            "index": i,
            "edge": edge,
            "edge_geom": edge_geom,
            "edge_length": edge_length,
            "points": points,  # Nx3 array
            "faces_of_edge" : [],
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
                edge_data['faces_of_edge'].append((face_data['index'], face_to_index_map[adj_face]))

                adj_index = face_to_index_map[adj_face]
                if edge_type == "Convex":
                    face_data['convex_adjacent'].append(adj_index)
                elif edge_type == "Concave":
                    face_data['concave_adjacent'].append(adj_index)
                elif edge_type == "Tangent":
                    face_data['tangent_adjacent'].append(adj_index)

    # 5. FINAL PASS: Determine Stock Faces
    # =========================================================
    # Now that 'concave_adjacent' is populated, we can check it safely.

    for face_data in face_data_list:
        face_data["stock_face"] = define_stock_face(face_data)

    return all_faces, face_data_list, analyser, all_edges, edge_data_list


def print_face_analysis_table(all_faces, face_data_list):
    print("\n--- Model Face Analysis Table ---")
    print(f"Total faces found: {len(all_faces)}")
    print("-------------------------------------------------------------------")
    print(f"{'Face #':<3} | {'Type':<8} |{'Stock Face':<10} | {'Normal':<6} | {'Adjacent':<15} | {'Convex':<15} | {'Concave':<15} | {'Tangent'}")
    print("-" * 95)

    for face_data in face_data_list:
        print(f"{face_data['index']:<3} | "
              f"{face_data['type']:<10} | "
              f"{face_data['stock_face']:<10} | "
              f"{face_data['normal_vector_axis']:<6} | "
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



