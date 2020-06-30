class BaseError(Exception):
    """ Base Package Error"""


class InvalidModelInputError(BaseError):
    """Model input contains error"""