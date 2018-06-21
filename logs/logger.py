import sys, codecs
from logs.default_logger import DefaultLogger
from logs.debug_logger import DebugLogger
from logs.production_logger import ProductionLogger


class Logger(object):
    '''
    First create a speciific singelton Logger, e.g.

    ProductionLogger(filename=...)

    Then you can easily log your events simply by calling:

    Logger().log_error(msg)
    '''

    instance = None

    def __new__(cls, *args, **kwargs):
        if Logger.instance is None:
            Logger.instance = object.__new__(cls, *args, **kwargs)
        return Logger.instance

    def __init__(self, filename=None, slack_channel=None, slack_token=None):
        self.filename = filename
        self.slack_channel = slack_channel
        self.slack_token = slack_token

    @staticmethod
    def initialize(config):
        filename = config.log_path
        slack_channel = config.slack_channel
        slack_token = config.slack_token
        logger = {
            'default': DefaultLogger,
            'debug': DebugLogger,
            'production': ProductionLogger
        }[config.logger]
        logger(filename=filename, slack_channel=slack_channel, slack_token=slack_token)

    def log_debug(self, msg):
        raise TypeError('This method needs to be defined in subclass.')

    def log_error(self, msg):
        raise TypeError('This method needs to be defined in subclass.')

    def log_message(self, msg):
        raise TypeError('This method needs to be defined in subclass.')

    def log_result(self, msg):
        raise TypeError('This method needs to be defined in subclass.')

    def log_critical(self, msg):
        raise TypeError('This method needs to be defined in subclass.')

    def stdout(self, msg):
        print str(msg)

    def stderr(self, msg):
        sys.stderr.write(str(msg))

    def file(self, msg):
        with codecs.open(self.filename, 'a', encoding='utf-8') as f:
            f.write(str(msg))

    def slack(self, msg):
        from slackclient import SlackClient
        sc = SlackClient(self.slack_token)
        sc.api_call(
            "chat.postMessage",
            channel=self.slack_channel,
            text=str(msg)
        )