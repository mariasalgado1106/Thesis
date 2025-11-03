import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.SimpleGui import init_display
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core import TopoDS

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import GeomAbs_Plane, GeomAbs_Cylinder
from OCC.Core.TopoDS import TopoDS_Face
from OCC.Core.Geom import Geom_CylindricalSurface, Geom_Plane

from OCC.Core.BRepOffset import BRepOffset_Analyse
from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape




#1. Read STEP File
def load_step_file(filepath):

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

#2. Face Type
def get_face_geometry_type(face: TopoDS_Face):
    surf_adaptor = BRepAdaptor_Surface(face, True)
    face_type = surf_adaptor.GetType()

    if face_type == GeomAbs_Cylinder:
        return "Cylinder", surf_adaptor.Cylinder()
    elif face_type == GeomAbs_Plane:
        return "Plane", surf_adaptor.Plane()
    else:
        # You can add more types like GeomAbs_Sphere, GeomAbs_Torus etc.
        return "Other", None


#3.Adjacent Faces

#a target face of the all faces list is analysed
def get_adjacent_faces(target_face, all_faces_list):
    adjacent_faces = [] #create a list of the adj of that target face


    target_edges = set() #initialize a list of the target edges of the target face
    edge_explorer = TopExp_Explorer(target_face, TopAbs_EDGE) #explore if the edge corresponds to face
    while edge_explorer.More():
        target_edges.add(edge_explorer.Current()) #adds to the target edges list
        edge_explorer.Next() #proceeds to next one

    # Check edges of the other faces
    for face in all_faces_list:
        if face.IsSame(target_face): #if it's the same face as target, skip
            continue

        # Get edges of the other faces
        other_edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while other_edge_explorer.More():
            other_edge = other_edge_explorer.Current() #list of the edges of the other face

            #Check if any edge is shared
            for target_edge in target_edges:
                if target_edge.IsSame(other_edge):
                    adjacent_faces.append(face)
                    break  # Move to the next face

            if face in adjacent_faces:  # Optimization
                break

            other_edge_explorer.Next()

    return adjacent_faces

#4. Concave, complex...
def classify_edge_type(face1, face2, shared_edge, analyser):
    """Returns 'Convex', 'Concave', or 'Tangent' for a shared edge."""
    from OCC.Core.TopTools import TopTools_ListOfShape, TopTools_ListIteratorOfListOfShape

    # Get edges of each type for face1
    convex_edges = TopTools_ListOfShape()
    concave_edges = TopTools_ListOfShape()
    tangent_edges = TopTools_ListOfShape()

    analyser.Edges(face1, 0, concave_edges)  # ChFiDS_Concave = 0
    analyser.Edges(face1, 1, convex_edges)  # ChFiDS_Convex = 1
    analyser.Edges(face1, 2, tangent_edges)  # ChFiDS_Tangential = 2

    # Conversion to python list
    def occ_list_to_python(occ_list):
        """Convert TopTools_ListOfShape to Python list."""
        result = []
        iterator = TopTools_ListIteratorOfListOfShape(occ_list)
        while iterator.More():
            result.append(iterator.Value())
            iterator.Next()
        return result

    # Check which category the shared edge belongs to
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


#5. Feature Recognition

def find_features(all_faces, face_data_list):

    print("\n--- Starting Feature Recognition ---")

    recognized_features = []
    processed_faces = set()  # avoid double-counting

    # Through Hole
    for face_data in face_data_list:

        face = face_data["face"]
        if face in processed_faces:
            continue #avoid double counting

        # HINT 1: Is it a cylinder?
        if face_data["type"] == "Cylinder":
            print("Found a potential cylinder...")

            # HINT 2 & 3: Check adjacency
            # Get the data for all adjacent faces
            adj_face_indices = face_data["adjacent_indices"]

            planar_neighbors = 0
            for adj_index in adj_face_indices:
                adj_face_data = face_data_list[adj_index]
                if adj_face_data["type"] == "Plane":
                    planar_neighbors += 1 #check how many planar neighbors

            # Rule: A cylinder with at least 2 planar neighbors
            if planar_neighbors >= 2:
                print(">>> Found a 'Hole' feature!")

                feature_faces = [face]  # Start with the cylinder face

                # Add the adjacent planar faces to the feature
                for adj_index in adj_face_indices:
                    if face_data_list[adj_index]["type"] == "Plane":
                        feature_faces.append(face_data_list[adj_index]["face"])

                recognized_features.append(("Hole", feature_faces))

                # Mark all faces in this feature as processed
                for f in feature_faces:
                    processed_faces.add(f)

    # --- Add more rules for your 7 other features here ---
    # e.g., elif face_data["type"] == "Plane": ...

    return recognized_features


# MAIN BLOCK

if __name__ == "__main__":

    # --- STEP 1: LOAD & INITIALIZE ---

    display, start_display, add_menu, add_function_to_menu = init_display()

    file_path = os.path.join("STEPFiles", "example_thoughhole.stp")
    my_shape = load_step_file(file_path)
    # Initialize the BRepOffset_Analyse for the entire shape -> determine angles for edge analysis
    analyser = BRepOffset_Analyse(my_shape, 0.01)  # 0.01 is angle tolerance in radians

    if my_shape:
        # Display the base shape
        display.DisplayShape(my_shape, update=True, transparency=0.8, color="GRAY")

        # --- STEP 2: ANALYZE MODEL ---

        all_faces = []
        face_explorer = TopExp_Explorer(my_shape, TopAbs_FACE)
        while face_explorer.More():
            face = TopoDS.Face(face_explorer.Current())
            all_faces.append(face)
            face_explorer.Next()

        # Create a map for {face_object: index} for easy lookup
        face_to_index_map = {face: i for i, face in enumerate(all_faces)}

        # This list will hold all our data
        face_data_list = []

        # First pass: Get geometry type for all faces
        for i, face in enumerate(all_faces):
            face_type, geometry = get_face_geometry_type(face)
            face_data_list.append({
                "index": i,
                "face": face,
                "type": face_type,
                "geom": geometry,
                "adjacent_indices": []  # Initialize empty list
            })

        # Second pass: Get adjacency for all faces
        for face_data in face_data_list:
            adjacent_faces = get_adjacent_faces(face_data["face"], all_faces)
            # Store the *indices* of adjacent faces, not the objects
            adj_indices = [face_to_index_map[adj_f] for adj_f in adjacent_faces]
            face_data["adjacent_indices"] = adj_indices

        # Third pass: Classify edge types for adjacent faces
        for face_data in face_data_list:
            face_data["convex_adjacent"] = []
            face_data["concave_adjacent"] = []
            face_data["tangent_adjacent"] = []

            # Get edges of this face
            face_edges = []
            edge_explorer = TopExp_Explorer(face_data["face"], TopAbs_EDGE)
            while edge_explorer.More():
                face_edges.append(TopoDS.Edge(edge_explorer.Current()))
                edge_explorer.Next()

            # Track which adjacent faces we've already classified
            classified_faces = set()

            # For each adjacent face
            for adj_index in face_data["adjacent_indices"]:
                if adj_index in classified_faces:
                    continue  # Skip if already classified

                adj_face = face_data_list[adj_index]["face"]

                # Find shared edge
                adj_edge_explorer = TopExp_Explorer(adj_face, TopAbs_EDGE)
                shared_edge_found = False

                while adj_edge_explorer.More() and not shared_edge_found:
                    adj_edge = TopoDS.Edge(adj_edge_explorer.Current())

                    for face_edge in face_edges:
                        if face_edge.IsSame(adj_edge):
                            # Classify this edge
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

        # --- STEP 4: RUN FEATURE RECOGNITION & VISUALIZE ---

        # Pass the pre-computed data to the function for efficiency
        features = find_features(all_faces, face_data_list)

        print(f"\n--- Recognized {len(features)} features ---")

        # 5. Visualize the results
        colors = ["RED", "BLUE", "GREEN", "YELLOW", "ORANGE", "CYAN", "MAGENTA", "BLACK"]

        for i, (feature_name, faces) in enumerate(features):
            print(f"Feature {i + 1}: {feature_name} (composed of {len(faces)} faces)")

            # Display each face of the feature in a unique color
            color = colors[i % len(colors)]
            for face in faces:
                display.DisplayShape(face, update=False, color=color)

        # 6. Show the result
        display.FitAll()
        start_display()