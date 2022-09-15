class Steam(Exception):
    "Base exception for work with Steam API"


class SteamError(Steam):
    def __init__(self, status=None, data=None):
        self.status = status
        self.data = data

    def __str__(self):
        return f"Failed to retrieve data from the Steam API\nStatus code: {self.status}\nData: {self.data}"


class LiveGameNotFound(Steam):
    def __str__(self):
        return "Live game not found"


class PrizePoolNotFound(Steam):
    def __init__(self, status, leagueid):
        self.status = status
        self.leagueid = leagueid


    def __str__(self):
        return f"Leagueid: {self.leagueid} prize pool not found"
