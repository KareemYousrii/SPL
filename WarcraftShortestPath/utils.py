import os
import sys
import pickle
import random
import torch
import csv
from PIL import Image
import itertools
from collections import defaultdict, deque
import time
from functools import lru_cache
from constants import *

import ast
import collections
import json
from copy import deepcopy
from warnings import warn
import numpy as np

import inspect
import re
import shutil
import tempfile
from time import sleep

class customdefaultdict(defaultdict):
    def __missing__(self, key):
        if self.default_factory:
            dict.__setitem__(self, key, self.default_factory(key))
            return self[key]
        else:
            defaultdict.__missing__(self, key)

@lru_cache(maxsize=128)
def cached_np_load(path, **kwargs):
    return np.load(path, **kwargs)


def efficient_from_numpy(x, device):
    if device == 'cpu':
        return torch.from_numpy(x).cpu()
    else:
        return torch.from_numpy(x).contiguous().pin_memory().to(device=device, non_blocking=True)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name=None, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


def set_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)


def save_pickle(data, path):
    with open(path, "wb") as fh:
        pickle.dump(data, fh)


def load_pickle(path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def load_pngs(path):
    def get_im():
        for file in os.listdir(path):
            if file.endswith(".png"):
                im = Image.open(os.path.join(path, file))
                np_im = np.array(im)
                yield np_im

    return np.stack(list(get_im()))


def concat_2d(arr):
    rows, columns, channels, height, width = arr.shape
    return np.rollaxis(arr, 2, 0).swapaxes(2, 3).reshape(channels, height * rows, width * columns)


def sample_image_grid(height, width, images, labels):
    num_images = images.shape[0]
    num_labels = labels.shape[0]
    assert num_images == num_labels

    indices = np.random.choice(num_images, size=(height, width))
    return images[indices, ...], labels[indices]

class TrainingIterator(object):
    def __init__(self, data_dict):
        zipped_data = list(zip(*data_dict.values()))

        self.dtype = [(key, "f4", value[0].shape) for key, value in data_dict.items()]
        # PyTorch works with 32-bit floats by default

        self.array = np.array(zipped_data, dtype=self.dtype)

    def get_epoch_iterator(self, batch_size, number_of_epochs, device='cpu', preload=False, shuffle=True):
        def iterator():
            if preload:
                preload_deque = deque(maxlen=10)
            for i in range(number_of_epochs):
                if shuffle:
                    np.random.shuffle(self.array)
                for j in range(1 + len(self.array) // batch_size):
                    numpy_batch = self.array[j * batch_size : (j + 1) * batch_size]
                    torch_batch = {key: efficient_from_numpy(numpy_batch[key], device=device) for key in numpy_batch.dtype.names}

                    if numpy_batch.size:
                        if j == 0 and preload :
                            preload_deque.appendleft(torch_batch)
                            continue
                        if preload:
                            preload_deque.appendleft(torch_batch)
                            yield preload_deque.pop()
                        else:
                            yield torch_batch
                if preload:
                    while len(preload_deque) > 0:
                        yield preload_deque.pop()

        return iterator()

def detach_to_cpu_np(arrs):
    detached = [arr.cpu().detach().numpy() for arr in arrs]
    return detached


def grid_to_im_coordinate(grid_x, grid_y, grid_x_max, grid_y_max, im_width, im_height):
    x_spacing = im_width / grid_x_max
    im_x = x_spacing * (0.5 + grid_x)
    y_spacing = im_height / grid_y_max
    im_y = y_spacing * (0.5 + grid_y)
    return im_x, im_y, x_spacing, y_spacing


def maybe_parallelize(function, arg_list):
    if ray.is_initialized():
        ray_fn = ray.remote(function)
        return ray.get([ray_fn.remote(arg) for arg in arg_list])
    else:
        return [function(arg) for arg in arg_list]


def optimizer_from_string(optimizer_name):
    dct = {"Adam": torch.optim.Adam, "SGD": torch.optim.SGD}
    return dct[optimizer_name]


def cost_ratio_func(vertex_costs, true_paths, suggested_paths, e2i=None, valid_paths=None):

    sps = []
    tps = []
    e2i = e2i

    # The cost of the path is the sum of costs of unique edges on
    # that path
    for vc, tp, sp in zip(vertex_costs, true_paths, suggested_paths):
        vc = vc.flatten()
        sps += [vc[torch.unique(e2i[sp.bool()]) - 1].sum()]
        tps += [vc[torch.unique(e2i[tp.bool()]) - 1].sum()]

    suggested_paths_costs = torch.stack(sps)
    true_paths_costs = torch.stack(tps)

    cost_ratio = (suggested_paths_costs / true_paths_costs)

    # Invalid paths have a very high cost
    if valid_paths is not None:
        cost_ratio[~valid_paths] = 1e6

    return cost_ratio


def all_accuracies(vertex_costs, true_labels, suggested_labels, num_thresholds, e2i=None, valid_paths=None):
    num_examples = len(true_labels)
    meets_threshold = [0] * num_thresholds

    indiv = (true_labels == suggested_labels).sum()/(64*264)

    cost_ratios = cost_ratio_func(vertex_costs, true_labels, suggested_labels, e2i, valid_paths)
    for i in range(len(meets_threshold)):
        meets_threshold[i] = (cost_ratios -1.0 < 10.0**(-i - 1)).sum()

    threshold_dict = {f"below_{10. ** (1-i)}_percent_acc": val.item() / num_examples for i, val in enumerate(meets_threshold)}
    threshold_dict['indiv_acc'] = indiv 
    return threshold_dict

def shorten_string(string, max_len):
  if len(string) > max_len - 3:
    return '...' + string[-max_len + 3:]
  return string


def get_caller_file(depth=2):
  _, filename, _, _, _, _ = inspect.stack()[depth]
  return filename


def check_valid_name(string):
  pat = '[A-Za-z0-9_.-]*$'
  if type(string) is not str:
    raise TypeError(('Parameter \'{}\' not valid. String expected.'.format(string)))
  if string in RESERVED_PARAMS:
    raise ValueError('Parameter name {} is reserved'.format(string))
  if string.endswith(STD_ENDING):
    raise ValueError('Parameter name \'{}\' not valid.'
                     'Ends with \'{}\' (may cause collisions)'.format(string, STD_ENDING))
  if not bool(re.compile(pat).match(string)):
    raise ValueError('Parameter name \'{}\' not valid. Only \'[0-9][a-z][A-Z]_-.\' allowed.'.format(string))
  if string.endswith('.') or string.startswith('.'):
    raise ValueError('Parameter name \'{}\' not valid. \'.\' not allowed at start/end'.format(string))


def rm_dir_full(dir_name):
  sleep(0.5)
  if os.path.exists(dir_name):
    shutil.rmtree(dir_name, ignore_errors=True)

  # filesystem is sometimes slow to response
  if os.path.exists(dir_name):
    sleep(1.0)
    shutil.rmtree(dir_name, ignore_errors=True)

  if os.path.exists(dir_name):
    warn(f'Removing of dir {dir_name} failed')


def create_dir(dir_name):
  if not os.path.exists(dir_name):
    os.makedirs(dir_name)


def flatten_nested_string_dict(nested_dict, prepend=''):
  for key, value in nested_dict.items():
    if type(key) is not str:
      raise TypeError('Only strings as keys expected')
    if isinstance(value, dict):
      for sub in flatten_nested_string_dict(value, prepend=prepend + str(key) + OBJECT_SEPARATOR):
        yield sub
    else:
      yield prepend + str(key), value


def save_dict_as_one_line_csv(dct, filename):
  with open(filename, 'w') as f:
    writer = csv.DictWriter(f, fieldnames=dct.keys())
    writer.writeheader()
    writer.writerow(dct)


def get_sample_generator(samples, hyperparam_dict, distribution_list, extra_settings=None):
  if bool(hyperparam_dict) == bool(distribution_list):
    raise TypeError('Exactly one of hyperparam_dict and distribution list must be provided')
  if distribution_list and not samples:
    raise TypeError('Number of samples not specified')
  if distribution_list:
    ans = distribution_list_sampler(distribution_list, samples)
  elif samples:
    assert hyperparam_dict
    ans = hyperparam_dict_samples(hyperparam_dict, samples)
  else:
    ans = hyperparam_dict_product(hyperparam_dict)
  if extra_settings is not None:
    return itertools.chain(extra_settings, ans)
  else:
    return ans





def process_other_params(other_params, hyperparam_dict, distribution_list):
  if hyperparam_dict:
    name_list = hyperparam_dict.keys()
  else:
    name_list = [distr.param_name for distr in distribution_list]
  for name, value in other_params.items():
    check_valid_name(name)
    if name in name_list:
      raise ValueError('Duplicate setting \'{}\' in other params!'.format(name))
    if not any([isinstance(value, allowed_type) for allowed_type in PARAM_TYPES]):
      raise TypeError('Settings must from the following types: {}, not {}'.format(PARAM_TYPES, type(value)))
  nested_items = [(name.split('.'), value) for name, value in other_params.items()]
  return nested_to_dict(nested_items)


def validate_hyperparam_dict(hyperparam_dict):
  for name, option_list in hyperparam_dict.items():
    check_valid_name(name)
    if type(option_list) is not list:
      raise TypeError('Entries in hyperparam dict must be type list (not {}: {})'.format(name, type(option_list)))
    for item in option_list:
      if not any([isinstance(item, allowed_type) for allowed_type in PARAM_TYPES]):
        raise TypeError('Settings must from the following types: {}, not {}'.format(PARAM_TYPES, type(item)))


def hyperparam_dict_samples(hyperparam_dict, num_samples):
  validate_hyperparam_dict(hyperparam_dict)
  nested_items = [(name.split(OBJECT_SEPARATOR), options) for name, options in hyperparam_dict.items()]

  for i in range(num_samples):
    nested_samples = [(nested_path, random.choice(options)) for nested_path, options in nested_items]
    yield nested_to_dict(nested_samples)


def hyperparam_dict_product(hyperparam_dict):
  validate_hyperparam_dict(hyperparam_dict)

  nested_items = [(name.split(OBJECT_SEPARATOR), options) for name, options in hyperparam_dict.items()]
  nested_names, option_lists = zip(*nested_items)

  for sample_from_product in itertools.product(*list(option_lists)):
    yield nested_to_dict(zip(nested_names, sample_from_product))


def default_to_regular(d):
  if isinstance(d, defaultdict):
    d = {k: default_to_regular(v) for k, v in d.items()}
  return d


def nested_to_dict(nested_items):
  nested_dict = lambda: defaultdict(nested_dict)
  result = nested_dict()
  for nested_key, value in nested_items:
    ptr = result
    for key in nested_key[:-1]:
      ptr = ptr[key]
    ptr[nested_key[-1]] = value
  return default_to_regular(result)


def distribution_list_sampler(distribution_list, num_samples):
  for distr in distribution_list:
    distr.prepare_samples(howmany=num_samples)
  for i in range(num_samples):
    nested_items = [(distr.param_name.split(OBJECT_SEPARATOR), distr.sample()) for distr in distribution_list]
    yield nested_to_dict(nested_items)

from pathlib2 import Path
home = str(Path.home())

def mkdtemp(prefix='cluster_utils', suffix=''):
  new_prefix = prefix + ('' if not suffix else '-' + suffix + '-')
  return tempfile.mkdtemp(prefix=new_prefix, dir=os.path.join(home, '.cache'))


def temp_directory(prefix='cluster_utils', suffix=''):
  new_prefix = prefix + ('' if not suffix else '-' + suffix + '-')
  return tempfile.TemporaryDirectory(prefix=new_prefix, dir=os.path.join(home, '.cache'))


class ParamDict(dict):
  """ An immutable dict where elements can be accessed with a dot"""
  __getattr__ = dict.__getitem__

  def __delattr__(self, item):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __setattr__(self, key, value):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __setitem__(self, key, value):
    raise TypeError("Setting object not mutable after settings are fixed!")

  def __deepcopy__(self, memo):
    """ In order to support deepcopy"""
    return ParamDict([(deepcopy(k, memo), deepcopy(v, memo)) for k, v in self.items()])

  def __repr__(self):
    return json.dumps(self, indent=4, sort_keys=True)


def recursive_objectify(nested_dict):
  "Turns a nested_dict into a nested ParamDict"
  result = deepcopy(nested_dict)
  for k, v in result.items():
    if isinstance(v, collections.Mapping):
      result[k] = recursive_objectify(v)
  return ParamDict(result)


class SafeDict(dict):
  """ A dict with prohibiting init from a list of pairs containing duplicates"""
  def __init__(self, *args, **kwargs):
    if args and args[0] and not isinstance(args[0], dict):
      keys, _ = zip(*args[0])
      duplicates =[item for item, count in collections.Counter(keys).items() if count > 1]
      if duplicates:
        raise TypeError("Keys {} repeated in json parsing".format(duplicates))
    super().__init__(*args, **kwargs)


def load_json(file):
  """ Safe load of a json file (doubled entries raise exception)"""
  with open(file, 'r') as f:
    data = json.load(f, object_pairs_hook=SafeDict)
  return data


def update_recursive(d, u, defensive=False):
  for k, v in u.items():
    if defensive and k not in d:
      raise KeyError("Updating a non-existing key")
    if isinstance(v, collections.Mapping):
      d[k] = update_recursive(d.get(k, {}), v)
    else:
      d[k] = v
  return d


def save_settings_to_json(setting_dict, model_dir):
  filename = os.path.join(model_dir, JSON_SETTINGS_FILE)
  with open(filename, 'w') as file:
    file.write(json.dumps(setting_dict, sort_keys=True, indent=4))


def save_metrics_params(metrics, params, save_dir=None):
  if save_dir is None:
    save_dir = params.model_dir
  create_dir(save_dir)
  save_settings_to_json(params, save_dir)

  param_file = os.path.join(save_dir, CLUSTER_PARAM_FILE)
  flattened_params = dict(flatten_nested_string_dict(params))
  save_dict_as_one_line_csv(flattened_params, param_file)

  time_elapsed = time.time() - update_params_from_cmdline.start_time
  if 'time_elapsed' not in metrics.keys():
    metrics['time_elapsed'] = time_elapsed
  else:
    warn('\'time_elapsed\' metric already taken. Automatic time saving failed.')
  metric_file = os.path.join(save_dir, CLUSTER_METRIC_FILE)
  save_dict_as_one_line_csv(metrics, metric_file)


def is_json_file(cmd_line):
  try:
    return os.path.isfile(cmd_line)
  except Exception as e:
    warn('JSON parsing suppressed exception: ', e)
    return False


def is_parseable_dict(cmd_line):
  try:
    res = ast.literal_eval(cmd_line)
    return isinstance(res, dict)
  except Exception as e:
    warn('Dict literal eval suppressed exception: ', e)
    return False


def update_params_from_cmdline(cmd_line=None, default_params=None, custom_parser=None, verbose=True):
  """ Updates default settings based on command line input.

  :param cmd_line: Expecting (same format as) sys.argv
  :param default_params: Dictionary of default params
  :param custom_parser: callable that returns a dict of params on success
  and None on failure (suppress exceptions!)
  :param verbose: Boolean to determine if final settings are pretty printed
  :return: Immutable nested dict with (deep) dot access. Priority: default_params < default_json < cmd_line
  """
  if not cmd_line:
    cmd_line = sys.argv

  if default_params is None:
    default_params = {}

  if len(cmd_line) < 2:
    cmd_params = {}
  elif custom_parser and custom_parser(cmd_line):  # Custom parsing, typically for flags
    cmd_params = custom_parser(cmd_line)
  elif len(cmd_line) == 2 and is_json_file(cmd_line[1]):
    cmd_params = load_json(cmd_line[1])
  elif len(cmd_line) == 2 and is_parseable_dict(cmd_line[1]):
    cmd_params = ast.literal_eval(cmd_line[1])
  else:
    raise ValueError('Failed to parse command line')

  update_recursive(default_params, cmd_params)

  if JSON_FILE_KEY in default_params:
    json_params = load_json(default_params[JSON_FILE_KEY])
    if 'default_json' in json_params:
      json_base = load_json(json_params[JSON_FILE_KEY])
    else:
      json_base = {}
    update_recursive(json_base, json_params)
    update_recursive(default_params, json_base)

  update_recursive(default_params, cmd_params)
  final_params = recursive_objectify(default_params)
  if verbose:
    print(final_params)

  update_params_from_cmdline.start_time = time.time()
  return final_params

update_params_from_cmdline.start_time = None
