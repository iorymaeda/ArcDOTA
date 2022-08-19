import yaml
import copy
import pickle
import pathlib


class PathBase:
    def _get_curernt_path(self) -> pathlib.Path:
        s=''
        for f in __file__.split('\\')[:-1]:
            s += (f + '\\') 
        return pathlib.Path(s)


class ConfigBase(PathBase):
    RADIANT_SIDE = [0, 1, 2, 3, 4]
    DIRE_SIDE = [128, 129, 130, 131, 132]
    # This is used in development, when we want load custom config, not from folder
    _configs = {}

    def _get_config(self, config_name: str) -> dict:
        if not config_name.endswith('.yaml'):
            config_path = config_name + '.yaml'
        else:
            config_path = copy.copy(config_name)
            config_name = config_name.replace('.yaml', '')

        if config_name in self._configs:
            return self._configs[config_name]

        # Go to configs folder
        path = self._get_curernt_path()
        path = path.parent.absolute()
        path = path.joinpath("configs/")
        with open(path / config_path, 'r') as stream:
            return yaml.safe_load(stream)


class SaveLoadBase:
    def _save(self, path:str):
        "Save states to path"
        with open(path, 'wb') as f:
            pickle.dump(self.__dict__, f)
            
    def _load(self, path:str):
        "Load states from path"
        with open(path, 'rb') as f:
            saved_obj: dict = pickle.load(f)

        for k, v in saved_obj.items():
            setattr(self, k, v)
