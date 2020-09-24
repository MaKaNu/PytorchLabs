class Error(Exception):
    """ Base class for all Exceptions """
    pass


class LoadingError(Error):
    """ Raised if something went wrong by loading the Dataset

    Attributes:
        value   -- value which caused the Error
    """

    def __init__(self, value) -> None:
        self.value = value
        msg = "The Dataset could not be loaded because of value: " + str(self.value)
        super().__init__(msg)