import os

def validate_path(path):
    if not path:
        return None
    normalized_path = os.path.normpath(path)
    if '\0' in normalized_path:
        return None
    if any(part == '..' for part in normalized_path.split(os.sep)):
        return None
    if not os.path.exists(normalized_path):
        return None
    return normalized_path
