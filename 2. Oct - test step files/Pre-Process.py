from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Display.SimpleGui import init_display

# Initialize the display
display, start_display, add_menu, add_function_to_menu = init_display()

# Create a reader object
reader = STEPControl_Reader()

# Read the STEP file
status = reader.ReadFile('1.test.stp')

if status == 1:  # Check if the file was read successfully
    reader.TransferRoots()
    shape = reader.OneShape()

    # Display the shape
    display.DisplayShape(shape, update=True)
    start_display()
else:
    print("Error: Could not read the STEP file.")

#expand on this to traverse the shape object and extract the face and edge data needed for your graph