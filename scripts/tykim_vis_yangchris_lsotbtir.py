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
                "folder_name": "samurai_TIR",
                
                "SAMURAI-T":{
                    "LSOTB-TIR": "samurai_TIR_tiny"
                },
                "SAMURAI-S":{
                    "LSOTB-TIR": "samurai_TIR_small"
                },
                "SAMURAI-B":{
                    "LSOTB-TIR": "samurai_TIR_base_plus"
                },
                "SAMURAI-L":{
                    "LSOTB-TIR": "samurai_TIR_large"
                }
            }
        }

    def get_tracker_folder(self, tracker_name, tracker_variant, dataset_name):
        assert tracker_name in self.vot_results_path, f"Tracker {tracker_name} results are not available."
        assert tracker_variant  in self.vot_results_path[tracker_name], f"Tracker variant {tracker_variant} is not available for tracker {tracker_name}."
        assert dataset_name in self.vot_results_path[tracker_name][tracker_variant], f"Dataset {dataset_name} is not available for tracker {tracker_name} variant {tracker_variant}."

        return self.vot_results_path[tracker_name]["folder_name"], self.vot_results_path[tracker_name][tracker_variant][dataset_name]

    def evaluate(self, dataset_name):
        trackers = []

        full_tracker_to_evaluate = {
            "LSOTB-TIR": {                
                "SAMURAI": ["SAMURAI-T", "SAMURAI-S", "SAMURAI-B", "SAMURAI-L"],
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
        elif dataset_name == "LSOTB-TIR":
            dataset = get_dataset('lsotb_tir_subset')
        dataset_name = "LSOTB-TIR"

        print_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))
        print_per_sequence_results(trackers, dataset, dataset_name, merge_results=True)
        plot_results(trackers, dataset, dataset_name, merge_results=True, plot_types=('success', 'prec', 'norm_prec'))


def main():
    vot_eval = VOTEval()

    # vot_eval.evaluate(dataset_name="LaSOT")
    vot_eval.evaluate(dataset_name="LSOTB-TIR")

if __name__ == '__main__':
    main()