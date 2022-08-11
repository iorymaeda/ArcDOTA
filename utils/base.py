import yaml
import pathlib


class ConfigBase:
    RADIANT_SIDE = [0, 1, 2, 3, 4]
    DIRE_SIDE = [128, 129, 130, 131, 132]

    def get_config(self, config_name: str) -> dict:
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

