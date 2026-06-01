"""Serialization encoders for internal toolkit types (JSON and YAML)."""

import json
import yaml
import collections
import datetime
from typing import Any
import numpy as np

from vot.utilities.data import Grid

class JSONEncoder(json.JSONEncoder):
    """JSON encoder for internal types."""

    def default(self, o: Any) -> Any:
        """Default encoder."""
        if isinstance(o, Grid):
            return list(o)
        elif isinstance(o, datetime.date):
            return o.strftime('%Y/%m/%d')
        elif isinstance(o, np.ndarray):
            return o.tolist()
        else:
            return super().default(o)

class YAMLEncoder(yaml.Dumper):
    """YAML encoder for internal types."""

    def represent_tuple(self, data: Any) -> Any:
        """Represents a tuple."""
        return self.represent_list(list(data))


    def represent_custom(self, data: Any) -> Any:
        """Represents a custom internal type."""
        if isinstance(data, Grid):
            return self.represent_list(list(data))
        elif isinstance(data, datetime.date):
            return self.represent_scalar('tag:yaml.org,2002:str', data.strftime('%Y/%m/%d'))
        elif isinstance(data, np.ndarray):
            return self.represent_list(data.tolist())
        else:
            return super().represent_object(data)

YAMLEncoder.add_representer(collections.OrderedDict, YAMLEncoder.represent_dict)
YAMLEncoder.add_representer(tuple, YAMLEncoder.represent_tuple)
YAMLEncoder.add_representer(Grid, YAMLEncoder.represent_custom)
YAMLEncoder.add_representer(np.ndarray, YAMLEncoder.represent_custom)
YAMLEncoder.add_multi_representer(np.integer, lambda dumper, data: dumper.represent_int(int(data)))
YAMLEncoder.add_multi_representer(np.inexact, lambda dumper, data: dumper.represent_float(float(data)))