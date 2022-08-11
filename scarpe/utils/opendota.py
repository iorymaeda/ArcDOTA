from requests import session
from .requests import OpendotaSession



class OpendotaWrapper:
    URL = {
        'league': "https://api.opendota.com/api/leagues/"
    }

    def __init__(self):
        self.session = OpendotaSession()


    async def leagues(self) -> dict:
        json, status = await self.session.get(self.URL['league'])
        assert status//100 == 2, f'status code: {status}'
        return json
            

