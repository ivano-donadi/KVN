import imp
import os


def _evaluator_factory(cfg, **kwargs):
    task = cfg.task
    module = '.'.join(['lib.evaluators', 'custom', task])
    path = os.path.join('lib/evaluators', 'custom', task+'.py')
    evaluator = imp.load_source(module, path).Evaluator(cfg.result_dir, **kwargs)
    return evaluator


def make_evaluator(cfg, **kwargs):
    if cfg.skip_eval:
        return None
    else:
        return _evaluator_factory(cfg, **kwargs)
