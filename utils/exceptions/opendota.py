class Opendota(Exception):
    "Base exception for work with OpenDota API"


class GameNotParsed(Opendota):
    def __str__(self):
        return "Game has not parsed, parse it with post request.\nLeague games parses automaticly.\nGame parsing may take up to 15 minutes"


class OpendotaError(Opendota):
    def __init__(self, status=None, data=None):
        self.status = status
        self.data = data

    def __str__(self):
        return f"Failed to retrieve data from the opendota API\nStatus code: {self.status}\nData: {self.data}"


class ParsingNotPossible(Opendota):
    def __str__(self):
        return "The game is not parsed and has passed a lot of time it will not be parsed"


class OpendotaMonthLimit(Opendota):
    def __str__(self):
        return "Month limit is over"


class OpendotaMinuteLimit(Opendota):
    def __str__(self):
        return "Minute limit is over"
