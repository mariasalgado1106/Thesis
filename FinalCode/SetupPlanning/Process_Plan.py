from FeatureRecognition.feature_recognition import FeatureRecognition
from FeatureRecognition.geometry_analysis import analyze_shape
from SetupPlanning.TAD_and_Dependencies import TAD_Extraction, Dependencies

class Process_Plan:
    def __init__(self, my_shape):
        self.shape = my_shape
        (self.all_faces, self.face_data_list, self.analyser, self.all_edges,
         self.edge_data_list) = analyze_shape(self.shape)

        self.recognizer = FeatureRecognition(self.shape)
        self.features = self.recognizer.identify_features()

        self.tad_extractor = TAD_Extraction(self.shape)
        self.dep_extractor = Dependencies(self.shape)
        self.feature_info = self.dep_extractor.identify_relationships()

    def group_by_tads(self):
        groups = {}
        for feat in self.feature_info:
            tads = feat.get('tads', [])
            if not tads:
                if "INACCESSIBLE" not in groups:
                    groups["INACCESSIBLE"] = []
                groups["INACCESSIBLE"].append(feat)
                continue
            for t in tads:
                axis = t['axis']
                if axis not in groups:
                    groups[axis] = []
                if feat not in groups[axis]:
                    groups[axis].append(feat)
        return groups

    def print_grouped_tads_and_dependencies(self):
        groups = self.group_by_tads()

        print("\n" + "=" * 80)
        print(f"{'SETUP (TAD)':<15} | {'FEATURE IDs':<30} | {'DEPENDENCIES'}")
        print("-" * 80)

        for axis in sorted(groups.keys()):
            feats = groups[axis]
            feat_ids = [str(f['feat_idx']) for f in feats]

            for f in feats:
                deps = ", ".join(map(str, f['dependency'])) if f['dependency'] else "None"
                # We print one row per feature in the setup group
                print(f"{axis:<15} | Feature {f['feat_idx']:<22} | Needs: {deps}")

            print("-" * 80)

    def filter_tads (self):
        #check the tad with most features -> this will be priority
        # see if within this priority tad there is any feature with dependency -> if yes, remove this
        #??i dont want to make smth definite, it should iterate through this in some way because i have the 2 optimization methods
        return None