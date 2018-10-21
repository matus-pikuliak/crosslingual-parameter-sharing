import sys, codecs


class Logger(object):

    def __init__(self, filename=None, slack_channel=None, slack_token=None):
        self.filename = filename
        self.slack_channel = slack_channel
        self.slack_token = slack_token

    def log_debug(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def log_error(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def log_message(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def log_result(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def log_critical(self, msg):
        raise NotImplementedError('This method needs to be defined in subclass.')

    def stdout(self, msg):
        print(str(msg))

    def stderr(self, msg):
        sys.stderr.write(str(msg))

    def file(self, msg):
        with codecs.open(self.filename, 'a', encoding='utf-8') as f:
            f.write(str(msg)+'\n')

    def slack(self, msg):
        from slackclient import SlackClient
        sc = SlackClient(self.slack_token)
        sc.api_call(
            "chat.postMessage",
            channel=self.slack_channel,
            text=str(msg)
        )
