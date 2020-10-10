import os
import json
import yaml


def read_json(json_path):
    assert json_path, f'{json_path} not exist'
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data


class Loader(yaml.SafeLoader):
    def __init__(self, stream):

        self._root = os.path.split(stream.name)[0]

        super(Loader, self).__init__(stream)

    def include(self, node):

        filename = os.path.join(self._root, self.construct_scalar(node))

        with open(filename, 'r') as f:
            return yaml.load(f, Loader)

# enable PyYAML to handle "!include"
Loader.add_constructor('!include', Loader.include)


def read_yaml(yaml_path):
    assert yaml_path, f'{yaml_path} not exist'
    with open(yaml_path, 'r') as f:
        data = yaml.load(f, Loader = Loader)
    return data