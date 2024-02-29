import os
import imp
from lib.datasets.dataset_catalog import DatasetCatalog


def make_visualizer(cfg):
    task = cfg.task
    module = '.'.join(['lib.visualizers', 'custom', task])
    path = os.path.join('lib/visualizers', 'custom', task+'.py')
    visualizer = imp.load_source(module, path).Visualizer()
    return visualizer
