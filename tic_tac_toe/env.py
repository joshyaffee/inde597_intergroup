from ..environments import EnvironmentVersus, Agent

class TicTacToeEnvironment(EnvironmentVersus):
    current_state = [0] * 9
    previous_state = None