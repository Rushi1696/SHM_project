"""Data validation helpers."""


def validate_data(record):
    """Validate data record"""
    if record is None:
        return False
    return True


def validate(record):
    """Alias for validate_data"""
    return validate_data(record)
