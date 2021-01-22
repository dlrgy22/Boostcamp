import logging

if __name__ == "__main__":
    logger = logging.getLogger("main")
    logging.basicConfig(level = logging.DEBUG)

    #파일에 저장
    stream_handler = logging.FileHandler(
        "my.log", mode="a", encoding="utf8")
    logger.addHandler(stream_handler)

    logger.debug("dubug")
    logger.info("info")
    logger.warning("warning")
    logger.error("error")
    logger.critical("critical")