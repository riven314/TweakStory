CONFIG_FILE = './config/app_config.yaml'

#HOST = 'http://127.0.0.1' # when API server is local and not in Docker mode
HOST = 'http://fastapi' # when API server is in Docker mode
PORT = 8080
ROUTE = '/inference'

PADDING_CODE = '&nbsp;'
DEFAULT_PADDING = PADDING_CODE * 10

SENTENCE_CLASS_MAP = {'Short Caption': 0, 'Mid Caption': 1, 'Long Caption': 2}
EMOJI_CLASS_MAP = {'With Emoji': 1, 'Without Emoji': 0}