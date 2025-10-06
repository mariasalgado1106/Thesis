from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import (TopAbs_SOLID, TopAbs_SHELL, TopAbs_FACE,
                             TopAbs_WIRE, TopAbs_EDGE, TopAbs_VERTEX)
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface, BRepAdaptor_Curve
from OCC.Core.GeomAbs import (GeomAbs_Plane, GeomAbs_Cylinder, GeomAbs_Cone,
                              GeomAbs_Sphere, GeomAbs_Torus, GeomAbs_BezierSurface,
                              GeomAbs_BSplineSurface, GeomAbs_Line, GeomAbs_Circle,
                              GeomAbs_Ellipse, GeomAbs_Hyperbola, GeomAbs_Parabola,
                              GeomAbs_BezierCurve, GeomAbs_BSplineCurve)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepGProp import brepgprop_VolumeProperties, brepgprop_SurfaceProperties
import sys


def read_step_file(filename):
    """Read STEP file and return the shape"""
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)

    if status != IFSelect_RetDone:
        raise Exception(f"Error reading STEP file: {filename}")

    step_reader.TransferRoots()
    shape = step_reader.OneShape()
    return shape


def get_surface_type(face):
    """Determine the type of surface"""
    surf = BRepAdaptor_Surface(face)
    surf_type = surf.GetType()

    type_map = {
        GeomAbs_Plane: "Plane",
        GeomAbs_Cylinder: "Cylinder",
        GeomAbs_Cone: "Cone",
        GeomAbs_Sphere: "Sphere",
        GeomAbs_Torus: "Torus",
        GeomAbs_BezierSurface: "Bezier Surface",
        GeomAbs_BSplineSurface: "B-Spline Surface"
    }

    return type_map.get(surf_type, "Other Surface")


def get_curve_type(edge):
    """Determine the type of curve"""
    curve = BRepAdaptor_Curve(edge)
    curve_type = curve.GetType()

    type_map = {
        GeomAbs_Line: "Line",
        GeomAbs_Circle: "Circle",
        GeomAbs_Ellipse: "Ellipse",
        GeomAbs_Hyperbola: "Hyperbola",
        GeomAbs_Parabola: "Parabola",
        GeomAbs_BezierCurve: "Bezier Curve",
        GeomAbs_BSplineCurve: "B-Spline Curve"
    }

    return type_map.get(curve_type, "Other Curve")


def analyze_geometry(shape):
    """Analyze and count geometric features"""
    results = {
        'solids': 0,
        'shells': 0,
        'faces': 0,
        'wires': 0,
        'edges': 0,
        'vertices': 0,
        'surface_types': {},
        'curve_types': {}
    }

    # Count solids
    exp = TopExp_Explorer(shape, TopAbs_SOLID)
    while exp.More():
        results['solids'] += 1
        exp.Next()

    # Count shells
    exp = TopExp_Explorer(shape, TopAbs_SHELL)
    while exp.More():
        results['shells'] += 1
        exp.Next()

    # Count and classify faces
    exp = TopExp_Explorer(shape, TopAbs_FACE)
    while exp.More():
        results['faces'] += 1
        face = exp.Current()
        surf_type = get_surface_type(face)
        results['surface_types'][surf_type] = results['surface_types'].get(surf_type, 0) + 1
        exp.Next()

    # Count wires
    exp = TopExp_Explorer(shape, TopAbs_WIRE)
    while exp.More():
        results['wires'] += 1
        exp.Next()

    # Count and classify edges
    exp = TopExp_Explorer(shape, TopAbs_EDGE)
    while exp.More():
        results['edges'] += 1
        edge = exp.Current()
        curve_type = get_curve_type(edge)
        results['curve_types'][curve_type] = results['curve_types'].get(curve_type, 0) + 1
        exp.Next()

    # Count vertices
    exp = TopExp_Explorer(shape, TopAbs_VERTEX)
    while exp.More():
        results['vertices'] += 1
        exp.Next()

    return results


def calculate_properties(shape):
    """Calculate volume and surface area"""
    props = GProp_GProps()

    # Volume
    brepgprop_VolumeProperties(shape, props)
    volume = props.Mass()

    # Surface area
    brepgprop_SurfaceProperties(shape, props)
    surface_area = props.Mass()

    return volume, surface_area


def print_report(filename, results, volume, surface_area):
    """Print formatted analysis report"""
    print("=" * 70)
    print(f"STEP FILE ANALYSIS: {filename}")
    print("=" * 70)
    print()

    print("TOPOLOGICAL SUMMARY:")
    print("-" * 70)
    print(f"  Solids:    {results['solids']}")
    print(f"  Shells:    {results['shells']}")
    print(f"  Faces:     {results['faces']}")
    print(f"  Wires:     {results['wires']}")
    print(f"  Edges:     {results['edges']}")
    print(f"  Vertices:  {results['vertices']}")
    print()

    print("SURFACE TYPES:")
    print("-" * 70)
    if results['surface_types']:
        for surf_type, count in sorted(results['surface_types'].items()):
            print(f"  {surf_type:<20} : {count}")
    else:
        print("  No surfaces found")
    print()

    print("CURVE TYPES:")
    print("-" * 70)
    if results['curve_types']:
        for curve_type, count in sorted(results['curve_types'].items()):
            print(f"  {curve_type:<20} : {count}")
    else:
        print("  No curves found")
    print()

    print("GEOMETRIC PROPERTIES:")
    print("-" * 70)
    print(f"  Volume:        {volume:.6f} cubic units")
    print(f"  Surface Area:  {surface_area:.6f} square units")
    print()
    print("=" * 70)


def main():
    if len(sys.argv) < 2:
        print("Usage: python step_analyzer.py <step_file>")
        print("Example: python step_analyzer.py 1.test")
        sys.exit(1)

    filename = sys.argv[1]

    try:
        print(f"\nReading STEP file: {filename}...")
        shape = read_step_file(filename)

        print("Analyzing geometry...")
        results = analyze_geometry(shape)

        print("Calculating properties...")
        volume, surface_area = calculate_properties(shape)

        print("\n")
        print_report(filename, results, volume, surface_area)

    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()