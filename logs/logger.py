import codecs
import os

from constants import LOG_CRITICAL, LOG_RESULT, LOG_MESSAGE


class Logger(object):

    def __init__(self, server_name=None, filename=None, slack_channel=None, slack_token=None):
        self.server_name = server_name
        self.filename = filename
        self.slack_channel = slack_channel
        self.slack_token = slack_token

    @staticmethod
    def factory(type, **kwargs):
        from logs.debug_logger import DebugLogger
        from logs.default_logger import DefaultLogger
        from logs.production_logger import ProductionLogger
        logger_type = {
            'default': DefaultLogger,
            'debug': DebugLogger,
            'production': ProductionLogger
        }[type]
        return logger_type(**kwargs)

    def log(self, msg, level):
        f = {
            LOG_CRITICAL: self.log_critical,
            LOG_RESULT: self.log_result,
            LOG_MESSAGE: self.log_message
        }
        f[level](msg)

    def log_message(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def log_result(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def log_critical(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def stdout(self, msg):
        print(str(msg))

    def file(self, msg, mode='a'):
        with codecs.open(self.filename, mode=mode, encoding='utf-8') as f:
            f.write(str(msg)+'\n')

    def slack(self, msg):
        from slackclient import SlackClient
        sc = SlackClient(self.slack_token)
        sc.api_call(
            "chat.postMessage",
            channel=self.slack_channel,
            text=str(msg)
        )

    def system(self, msg):
        os.system(f'notify-send "{msg}"')
