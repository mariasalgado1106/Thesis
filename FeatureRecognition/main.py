import os
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Display.SimpleGui import init_display
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE, TopAbs_EDGE
from OCC.Core.BRep import BRep_Tool
from OCC.Core.Geom import Geom_CylindricalSurface, Geom_Plane
from OCC.Core import TopoDS


def load_step_file(filepath):
    """Loads a STEP file and returns the main shape."""
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


# -----------------------------------------------------------
# CHANGE 1: Replaced this entire function
# -----------------------------------------------------------
def get_face_geometry(face):
    """Returns the geometry (e.g., plane, cylinder) of a TopoDS_Face."""
    surface = BRep_Tool.Surface(face)  # This returns a Handle_Geom_Surface

    # First, check for a null handle (which evaluates to False)
    if not surface:
        return None

    # --- THIS IS THE NEW, ROBUST METHOD ---
    # We get the C++ type name of the surface as a string
    # This call is safe and does not crash.
    type_name = surface.DynamicType().Name()

    # Now we check the string name
    if type_name == "Geom_CylindricalSurface":
        # The type is correct, so we can safely downcast
        return Geom_CylindricalSurface.DownCast(surface)

    if type_name == "Geom_Plane":
        # The type is correct, so we can safely downcast
        return Geom_Plane.DownCast(surface)

    # ... add more types as needed
    # e.g., if type_name == "Geom_Cone":
    #           return Geom_Cone.DownCast(surface)

    return None  # Not a type we recognize


# -----------------------------------------------------------


# -----------------------------------------------------------
# CHANGE 2: Fixed PascalCase methods in this function
# -----------------------------------------------------------
def get_adjacent_faces(target_face, all_faces_list):
    """Finds all faces that share at least one edge with the target_face."""
    adjacent_faces = []

    # 1. Get all edges of the target_face
    edge_explorer = TopExp_Explorer(target_face, TopAbs_EDGE)
    target_edges = []
    while edge_explorer.More():
        target_edges.append(edge_explorer.Current())
        edge_explorer.Next()

    # 2. Check all other faces
    for face in all_faces_list:
        if face.IsSame(target_face):
            continue

        # 3. Get edges of the other face
        other_edge_explorer = TopExp_Explorer(face, TopAbs_EDGE)
        while other_edge_explorer.More():
            other_edge = other_edge_explorer.Current()

            # 4. Check if any edge is shared
            for target_edge in target_edges:
                if target_edge.IsSame(other_edge):
                    adjacent_faces.append(face)
                    break  # Move to the next face

            if face in adjacent_faces:  # Optimization
                break

            other_edge_explorer.Next()

    return adjacent_faces


# -----------------------------------------------------------


# -----------------------------------------------------------
# CHANGE 3: Fixed PascalCase methods in this function
# -----------------------------------------------------------
def find_features(shape):
    """Main feature recognition function."""

    all_faces = []
    face_explorer = TopExp_Explorer(shape, TopAbs_FACE)
    while face_explorer.More():
        face = TopoDS.Face(face_explorer.Current())
        all_faces.append(face)
        face_explorer.Next()

    print(f"Model has {len(all_faces)} faces.")

    recognized_features = []
    processed_faces = set()  # To avoid double-counting

    # --- Feature Rule: Simple Through Hole ---
    for face in all_faces:
        if face in processed_faces:
            continue

        geom = get_face_geometry(face)

        # HINT 1: Is it a cylinder?
        if geom and isinstance(geom, Geom_CylindricalSurface):
            print("Found a potential cylinder...")

            # HINT 2 & 3: Check adjacency
            adj_faces = get_adjacent_faces(face, all_faces)

            # This is a very simple rule. A real one would be more robust.
            # e.g., check that *two* adjacent faces are planes
            # and their normals are parallel to the cylinder axis.

            planar_neighbors = 0
            for adj_face in adj_faces:
                adj_geom = get_face_geometry(adj_face)
                if adj_geom and isinstance(adj_geom, Geom_Plane):
                    planar_neighbors += 1

            # Rule: A cylinder with at least 2 planar neighbors
            if planar_neighbors >= 2:
                print(">>> Found a 'Hole' feature!")

                # Let's refine this to only include the *planar* neighbors
                # This is still a simple example
                feature_faces = [face]
                for adj_face in adj_faces:
                    adj_geom = get_face_geometry(adj_face)
                    if adj_geom and isinstance(adj_geom, Geom_Plane):
                        feature_faces.append(adj_face)

                recognized_features.append(("Hole", feature_faces))

                # Mark all faces in this feature as processed
                for f in feature_faces:
                    processed_faces.add(f)  # <-- This was correctly fixed before

    # --- Add more rules for your 7 other features here ---
    # e.g., elif is_pocket_seed(face):

    return recognized_features


# -----------------------------------------------------------


# --- Update your main block ---
if __name__ == "__main__":
    # 1. Initialize the viewer
    display, start_display, add_menu, add_function_to_menu = init_display()

    # 2. Load your STEP file
    # Using os.path.join is safer
    file_path = os.path.join("STEPFiles", "example_thoughhole.stp")
    my_shape = load_step_file(file_path)

    if my_shape:
        # 3. Display the base shape
        display.DisplayShape(my_shape, update=True, transparency=0.8, color="GRAY")

        # ------------------------------------------------------------------
        # --- NEW CODE: Display all faces, their geometry, and adjacency ---
        # ------------------------------------------------------------------
        print("\n--- Model Face Analysis ---")

        # 1. Get all faces into a list first
        all_faces = []
        face_explorer = TopExp_Explorer(my_shape, TopAbs_FACE)
        while face_explorer.More():
            face = TopoDS.Face(face_explorer.Current())
            all_faces.append(face)
            face_explorer.Next()

        # 2. Create a "face-to-index" map for fast lookups
        # We need this to print the *index* of adjacent faces, not the object
        face_to_index_map = {face: i for i, face in enumerate(all_faces)}

        print(f"Total faces found: {len(all_faces)}")
        print("-------------------------------------------------------------------")
        # Added 'Adjacent Faces' column and adjusted spacing
        print(f"{'Face #':<6} | {'Geometry Type':<25} | {'Adjacent Faces'}")
        print("-------------------------------------------------------------------")

        # 3. Loop through the list of faces
        for face_index, face in enumerate(all_faces):

            # Get geometry name
            surface = BRep_Tool.Surface(face)
            geom_name = "Unknown (Null Surface)"
            if surface:
                geom_name = surface.DynamicType().Name()

                # Get adjacent faces
            adjacent_faces = get_adjacent_faces(face, all_faces)

            # Convert adjacent face objects to their indices using the map
            adj_face_indices = [face_to_index_map[adj_face] for adj_face in adjacent_faces]

            # Print the formatted table row
            print(f"{face_index:<6} | {geom_name:<25} | {adj_face_indices}")

        print("-------------------------------------------------------------------\n")
        # -----------------------------------------------------------
        # --- END OF NEW CODE ---
        # -----------------------------------------------------------

        # 4. Run the recognition
        features = find_features(my_shape)

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