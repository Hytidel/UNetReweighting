import logging


logger_obj = logging.getLogger(__name__)

logger_obj.setLevel(logging.DEBUG)
logger_obj.propagate = False

formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] %(name)s: %(message)s", 
    datefmt = "%Y-%m-%d %H:%M:%S"
)
handler = logging.StreamHandler()
handler.setFormatter(formatter)

if logger_obj.hasHandlers():
    logger_obj.handlers.clear()
logger_obj.addHandler(handler)

logger_obj.info("Logger initiated.")


def logger(
    msg: str, 
    log_type: str = "info"
):
    if log_type == "info":
        logger_obj.info(msg)
    elif log_type == "warning":
        logger_obj.warning(msg)
    elif log_type == "debug":
        logger_obj.debug(msg)
    elif log_type == "error":
        logger_obj.error(msg)
    elif log_type == "critical":
        logger_obj.critical(msg)
    else:
        raise NotImplementedError(f"Unsupported log type: {log_type}. ")
