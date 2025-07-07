from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path as osp
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt


def add_path(path):
    if path not in sys.path:
        sys.path.insert(0, path)

this_dir = osp.dirname(__file__)

prj_path = osp.join(this_dir, '..')
add_path(prj_path)

from lib.test.analysis.plot_results import (plot_results, print_results, 
                                             print_per_sequence_results, 
                                             print_per_attribute_results, 
                                             plot_per_attribute_results)
from lib.test.evaluation import get_dataset, trackerlist


mpl.rcParams.update(mpl.rcParamsDefault)
plt.rcParams['figure.figsize'] = [8, 8]


class VOTEval():
    def __init__(self):
        # you'll need to modify the local.py in vot-toolkit/lib/test/evaluation to run this script
        
        self.vot_results_path = {
            "SAMURAI":{
                "folder_name": "samurai",
                "SAMURAI-L":{
                    # "LaSOT": "samurai_base_plus",
                    # "LaSOT_ext": "samurai_base_plus"
                    "LaSOT": "samurai_large",
                    "LaSOT_ext": "samurai_large"
                }
            }
        }

    def get_tracker_folder(self, tracker_name, tracker_variant, dataset_name):
        assert tracker_name in self.vot_results_path, f"Tracker {tracker_name} results are not available."
        assert tracker_variant  in self.vot_results_path[tracker_name], f"Tracker variant {tracker_variant} is not available for tracker {tracker_name}."
        assert dataset_name in self.vot_results_path[tracker_name][tracker_variant], f"Dataset {dataset_name} is not available for tracker {tracker_name} variant {tracker_variant}."

        return self.vot_results_path[tracker_name]["folder_name"], self.vot_results_path[tracker_name][tracker_variant][dataset_name]

    def evaluate(self, dataset_name="LaSOT"):
        trackers = []

        full_tracker_to_evaluate = {
            "LaSOT_ext": {
                # "ROMTrack": ["ROMTrack-B"],
                # "EVPTrack": ["EVPTrack-B"],
                # "ODTrack": ["ODTrack-B", "ODTrack-L"],
                # "HIPTrack": ["HIPTrack-B"],
                # "AQATrack": ["AQATrack-B", "AQATrack-L"],
                # "LoRAT": ["LoRAT-B", "LoRAT-L"],
                # "SAM2.1": ["SAM2.1-T", "SAM2.1-S", "SAM2.1-B", "SAM2.1-L"],
                # "SAMURAI": ["SAMURAI-T", "SAMURAI-S", "SAMURAI-B", "SAMURAI-L"],
                # "OSTrack": ["OSTrack-B"],
                # "EVPTrack": ["EVPTrack-B"],
                # "ODTrack": ["ODTrack-L"],
                # "HIPTrack": ["HIPTrack-B"],
                # "AQATrack": ["AQATrack-L"],
                # "LoRAT": ["LoRAT-L"],
                # "SAM2.1": ["SAM2.1-L", "SAM2.1-B"],
                # "SAMURAI": ["SAMURAI-L", "SAMURAI-B"],
                
                "SAMURAI": ["SAMURAI-L"],
            }
        }
        tracker_to_evaluate = full_tracker_to_evaluate[dataset_name]

        for tracker_name, tracker_variants in tracker_to_evaluate.items():
            for tracker_variant in tracker_variants:
                tracker_folder, tracker_sub_folder = self.get_tracker_folder(tracker_name, tracker_variant, dataset_name)
                trackers.extend(trackerlist(tracker_folder, tracker_sub_folder, None, None, tracker_variant))

        if dataset_name == "LaSOT":
            dataset = get_dataset('lasot')
        elif dataset_name == "LaSOT_ext":
            dataset = get_dataset('lasot_extension_subset')
        dataset_name = "LaSOT$_{ext}$"

        print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
        plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))

            
def main():
    vot_eval = VOTEval()

    # vot_eval.evaluate(dataset_name="LaSOT")
    vot_eval.evaluate(dataset_name="LaSOT_ext")

if __name__ == '__main__':
    main()