import yaml
import copy
import json
import pickle
import pathlib
from contextlib import suppress


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


class DotaconstantsBase(PathBase):
    RADIANT_SIDE = [0, 1, 2, 3, 4]
    DIRE_SIDE = [128, 129, 130, 131, 132]

    def _get_constants_path(self) -> pathlib.Path:
        path = self._get_curernt_path()
        path = path.parent.resolve()
        path = path / "scarpe/dotaconstants/"
        return path.resolve()


    def _load_json(self, path: str | pathlib.Path) -> dict:
        for encoding in [None, 'utf8', 'utf16', 'utf32']:
            with (suppress(UnicodeDecodeError), open(path, 'r', encoding=encoding) as f):
                    return json.load(f)
        raise UnicodeDecodeError


    def _load_heroes(self) -> tuple[dict[str, int], dict[int, str]]:
        path = self._get_constants_path()
        path = path / "build/heroes.json"

        heroes = self._load_json(path)

        npc_to_id = {heroes[hero_id]['name']: int(hero_id) for hero_id in heroes}
        id_to_hero = {int(hero_id): heroes[hero_id]['localized_name'] for hero_id in heroes}
        return npc_to_id, id_to_hero

        
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
