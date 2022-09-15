######## Inference exceptions ########
class Inference(Exception):
    pass


class UntokenizedTeam(Inference):
    def __init__(self, team):
        self.team = team

    def __str__(self):
        return f"You tried to predict game with team ({self.team}) that has no it's own token"

class RadiantUntokenizedTeam(UntokenizedTeam):
    def __str__(self):
        return f"You tried to predict game with radiant team ({self.team}) that has no it's own token"

class DireUntokenizedTeam(UntokenizedTeam):
    def __str__(self):
        return f"You tried to predict game with dire team ({self.team}) that has no it's own token"


class NotEnoughGames(Inference):
    def __init__(self, team):
        self.team = team

    def __str__(self):
        return f"Team: {self.team} does not have enough past games"

class RadiantNotEnoughGames(NotEnoughGames):
    def __str__(self):
        return f"Radiant team: {self.team} does not have enough past games"

class DireNotEnoughGames(NotEnoughGames):
    def __str__(self):
        return f"Dire team: {self.team} does not have enough past games"



class LeagueIDNotFound(Inference):
    def __init__(self, leagueid):
        self.leagueid = leagueid

    def __str__(self):
        return f"League id: {self.leagueid} not founded in leagues json"

class LeaguePPNotFound(Inference):
    def __init__(self, leagueid):
        self.leagueid = leagueid

    def __str__(self):
        return f"League id: {self.leagueid} not founded in leagues prize pools json"

class LeaguesJSONsNotFound(LeagueIDNotFound, LeaguePPNotFound):
    def __init__(self, leagueid):
        self.leagueid = leagueid

    def __str__(self):
        return f"League id: {self.leagueid} not founded in leagues json and leagues prize pools json"

