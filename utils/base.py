import yaml
import pickle
import pathlib

from typing import Any

class ConfigBase:
    RADIANT_SIDE = [0, 1, 2, 3, 4]
    DIRE_SIDE = [128, 129, 130, 131, 132]

    def _get_config(self, config_name: str) -> dict:
        # Go to configs folder
        path = pathlib.Path(self._get_curernt_path())
        path = path.parent.absolute()
        path = path.joinpath("configs/")

        if not config_name.endswith('.yaml'):
            config_name += '.yaml'

        with open(path / config_name, 'r') as stream:
            return yaml.safe_load(stream)

    def _get_curernt_path(self) -> str:
        s=''
        for f in __file__.split('\\')[:-1]:
            s += (f + '\\') 
        return s


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
