# a custom error to stand out for the menu

from utils.screen import error
class PilotError(Exception):
    def __init__(self, message='An unexpected error occured'):
        error(message)