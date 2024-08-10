from utils.clogging import getColoredLogger
from utils.initialize import initializer

logger = getColoredLogger(__name__)
logger.setLevel("INFO")
initializer(globals(), logger=logger)


