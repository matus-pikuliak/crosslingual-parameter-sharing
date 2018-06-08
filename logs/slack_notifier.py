from slackclient import SlackClient


class SlackNotifier:

    def __init__(self, token, channel):
        self.token = token
        self.channel = channel

    def send(self, msg):
        sc = SlackClient(self.token)
        sc.api_call(
            "chat.postMessage",
            channel=self.channel,
            text=msg
        )