class Hawk(Exception):
    "Base exception for work with Hawk"


class MatchNotFound(Hawk):
    def __str__(self):
        return "Hawk match not found or can't be load"

class MainPageNotFound(Hawk):
    def __str__(self):
        return "Hawk main page not found or can't be load"


class CantParseMatch(Hawk):
    def __str__(self):
        return "Cant parse hawk match for some reasons"

class CantParseSeries(Hawk):
    def __str__(self):
        return "Cant parse hawk match for some reasons"