import os
import re
import sys
import threading
import builtins
import logging
from datetime import datetime, timedelta
from logging.handlers import RotatingFileHandler

_LOG_FORMAT = '%(asctime)s-%(levelname)s-%(module)s-%(funcName)s-%(lineno)d-%(message)s'
_DEFAULT_LOG_LEVEL = logging.INFO
_DEFAULT_LOG_FILE = 'app.log'
_MAX_BYTES = 100 * 1024 * 1024
_BACKUP_COUNT = 6
_RETENTION_DAYS = 30
_PRINT_REDIRECT_INSTALLED = False
_PRINT_LOGGER_NAME = 'print_redirect'
_CONSOLE_REDIRECT_INSTALLED = False
_FD_REDIRECT_INSTALLED = False
_FD_REDIRECT_READERS = []
_FD_EMIT_STREAMS = {}

def _project_root():
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

def _logs_dir():
    path = os.path.join(_project_root(), 'logs')
    os.makedirs(path, exist_ok=True)
    return path

def _safe_move(src, dst):
    if not os.path.exists(src):
        return
    target = dst
    if os.path.exists(target):
        index = 1
        while os.path.exists(f'{dst}.{index}'):
            index += 1
        target = f'{dst}.{index}'
    os.replace(src, target)

def _archive_logs_by_date(logs_dir, base_log_name):
    archive_root = os.path.join(logs_dir, 'archive')
    os.makedirs(archive_root, exist_ok=True)
    for entry in os.listdir(logs_dir):
        full_path = os.path.join(logs_dir, entry)
        if not os.path.isfile(full_path):
            continue
        if entry == base_log_name:
            continue
        if not entry.startswith(base_log_name):
            continue
        modified_date = datetime.fromtimestamp(os.path.getmtime(full_path)).strftime('%Y-%m-%d')
        target_dir = os.path.join(archive_root, modified_date)
        os.makedirs(target_dir, exist_ok=True)
        _safe_move(full_path, os.path.join(target_dir, entry))

def _cleanup_expired_logs(logs_dir, retention_days=_RETENTION_DAYS):
    expire_time = datetime.now() - timedelta(days=retention_days)
    for root, _, files in os.walk(logs_dir):
        for file_name in files:
            full_path = os.path.join(root, file_name)
            try:
                modified = datetime.fromtimestamp(os.path.getmtime(full_path))
                if modified < expire_time:
                    os.remove(full_path)
            except OSError:
                continue

def set_log_level(level):
    if isinstance(level, str):
        level = getattr(logging, level.upper(), _DEFAULT_LOG_LEVEL)
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    for handler in root_logger.handlers:
        handler.setLevel(level)
    return level

def configure_logging(log_file_name=_DEFAULT_LOG_FILE, level=_DEFAULT_LOG_LEVEL):
    logs_dir = _logs_dir()
    _archive_logs_by_date(logs_dir, log_file_name)
    _cleanup_expired_logs(logs_dir)
    root_logger = logging.getLogger()
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)
    root_logger.setLevel(_DEFAULT_LOG_LEVEL)
    formatter = logging.Formatter(_LOG_FORMAT)
    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, log_file_name),
        maxBytes=_MAX_BYTES,
        backupCount=_BACKUP_COUNT,
        encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    if os.environ.get('KVD_CONSOLE_LOG', '1') == '1':
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        root_logger.addHandler(stream_handler)
    set_log_level(level)
    logging.captureWarnings(True)
    return root_logger

def _redirected_print(*args, **kwargs):
    sep = kwargs.get('sep', ' ')
    end = kwargs.get('end', '\n')
    message = sep.join(str(arg) for arg in args)
    if end != '\n':
        message += end
    text = message.rstrip('\n')
    if text:
        logging.getLogger(_PRINT_LOGGER_NAME).info(text)

_redirected_print.__name__ = 'print'
_redirected_print.__qualname__ = 'print'
_redirected_print.__module__ = 'builtins'
print = _redirected_print

def redirect_print_to_logger(logger_name='print_redirect'):
    global _PRINT_REDIRECT_INSTALLED, _PRINT_LOGGER_NAME
    if _PRINT_REDIRECT_INSTALLED:
        return
    _PRINT_LOGGER_NAME = logger_name
    builtins.print = _redirected_print
    _PRINT_REDIRECT_INSTALLED = True

def _looks_like_progress_output(text):
    if '\r' in text:
        return True
    stripped = text.strip()
    if not stripped:
        return False
    if 'it/s' in stripped:
        return True
    return ('%' in stripped and '|' in stripped)

def _extract_progress_fragments(text):
    if text is None:
        return []
    progress_pattern = re.compile(r'([^\r\n]*\d{1,3}%\|[^\r\n]*\|\s*\d+/\d+\s*\[[^\r\n]*?\])')
    fragments = []
    for match in progress_pattern.finditer(str(text)):
        fragment = match.group(1)
        if fragment:
            fragments.append(fragment)
    return fragments

def _split_progress_and_noise(text):
    output = str(text or '')
    progress_fragments = _extract_progress_fragments(output)
    noise = output
    for fragment in progress_fragments:
        noise = noise.replace(fragment, ' ')
    noise = noise.replace('\r', '\n')
    noise_lines = [line.strip() for line in noise.splitlines() if line.strip()]
    return progress_fragments, noise_lines

def _emit_progress_fragments(fragments, stream):
    if not fragments:
        return 0
    written = 0
    for fragment in fragments:
        written += stream.write('\r' + fragment)
    stream.flush()
    return written

def _log_noise_lines(noise_lines, logger_name):
    if not noise_lines:
        return
    logger = logging.getLogger(logger_name)
    for line in noise_lines:
        logger.info(line)

def _resolve_emit_stream(fd, fallback_stream):
    stream = _FD_EMIT_STREAMS.get(fd)
    if stream is not None:
        return stream
    return fallback_stream

def _start_fd_redirect(fd, logger_name, fallback_stream):
    original_fd = os.dup(fd)
    encoding = getattr(fallback_stream, 'encoding', None) or 'utf-8'
    emit_stream = os.fdopen(original_fd, 'w', encoding=encoding, errors='replace', buffering=1)
    _FD_EMIT_STREAMS[fd] = emit_stream
    read_fd, write_fd = os.pipe()
    os.dup2(write_fd, fd)
    os.close(write_fd)
    def _reader():
        with os.fdopen(read_fd, 'rb', buffering=0) as reader:
            while True:
                chunk = reader.read(4096)
                if not chunk:
                    break
                text = chunk.decode(encoding, errors='replace')
                progress_fragments, noise_lines = _split_progress_and_noise(text)
                _emit_progress_fragments(progress_fragments, emit_stream)
                _log_noise_lines(noise_lines, logger_name)
    worker = threading.Thread(target=_reader, daemon=True)
    worker.start()
    _FD_REDIRECT_READERS.append(worker)

class _ProgressOnlyConsoleStream:
    def __init__(self, stream, logger_name, fd=None):
        self._stream = stream
        self._logger_name = logger_name
        self._fd = fd
        self._last_progress = False
        self.encoding = getattr(stream, 'encoding', 'utf-8')

    def write(self, text):
        if text is None:
            return 0
        output = str(text)
        if not output:
            return 0
        target_stream = _resolve_emit_stream(self._fd, self._stream)
        progress_fragments, noise_lines = _split_progress_and_noise(output)
        if progress_fragments:
            combined = ''.join(('\r' + fragment) for fragment in progress_fragments)
            self._last_progress = True
            target_stream.write(combined)
            target_stream.flush()
            _log_noise_lines(noise_lines, self._logger_name)
            return len(output)
        is_progress = _looks_like_progress_output(output)
        if is_progress or ((output == '\n' or output == '\r\n') and self._last_progress):
            self._last_progress = is_progress
            return target_stream.write(output)
        cleaned = output.strip()
        if cleaned:
            logging.getLogger(self._logger_name).info(cleaned)
        self._last_progress = False
        return len(output)

    def flush(self):
        return self._stream.flush()

    def isatty(self):
        isatty = getattr(self._stream, 'isatty', None)
        if callable(isatty):
            return isatty()
        return False

    def fileno(self):
        fileno = getattr(self._stream, 'fileno', None)
        if callable(fileno):
            return fileno()
        raise OSError('fileno is not available')

def redirect_console_to_logger_allow_progress(logger_name='console_redirect'):
    global _CONSOLE_REDIRECT_INSTALLED, _FD_REDIRECT_INSTALLED
    if _CONSOLE_REDIRECT_INSTALLED:
        return
    stdout_stream = getattr(sys, '__stdout__', None) or sys.stdout
    stderr_stream = getattr(sys, '__stderr__', None) or sys.stderr
    if not _FD_REDIRECT_INSTALLED:
        _start_fd_redirect(1, logger_name, stdout_stream)
        _start_fd_redirect(2, logger_name, stderr_stream)
        _FD_REDIRECT_INSTALLED = True
    sys.stdout = _ProgressOnlyConsoleStream(stdout_stream, logger_name, fd=1)
    sys.stderr = _ProgressOnlyConsoleStream(stderr_stream, logger_name, fd=2)
    _CONSOLE_REDIRECT_INSTALLED = True

def get_logger(name='kolo'):
    if not logging.getLogger().handlers:
        configure_logging()
    return logging.getLogger(name)
