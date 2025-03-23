# logger_utils.py

_log_callback = print  # Domyślna funkcja logująca

def set_logger(logger_function):
    """Ustawia funkcję logującą."""
    global _log_callback
    _log_callback = logger_function

def log(msg):
    """Loguje wiadomość."""
    _log_callback(str(msg))
