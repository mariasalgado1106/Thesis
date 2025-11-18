from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_EDGE
from OCC.Core.TopoDS import TopoDS
from OCC.Display.OCCViewer import rgb_color

class PartVisualizer:
    def __init__(self, builder, recognizer):
        self.builder = builder
        self.recognizer = recognizer

    def visualize_edges(self):
        print("\nVisualizing Edge Types")
        processed_edges = set()

