class Hint_Filtering:

    def __init__(self):
        pass

    def _clear_hint(self, sent: str):
        if sent.find('Is there anything else you would like to know?') >= 0:
            sent = sent.replace('Is there anything else you would like to know?', '') \
                .replace('😊', '').replace(' .', '').strip()
        if sent.find('I hope these hints help you guess the answer!') >= 0:
            sent = sent.replace('I hope these hints help you guess the answer!', '') \
                .replace('😊', '').replace(' .', '').strip()
        if sent == '"':
            sent = ''
        if sent.find('😊') >= 0:
            sent = sent.replace('😊', '')
        if sent.startswith('.'):
            sent = ''
        sent = sent.strip()

        return sent

    def filtering(self, hints):
        cleared_hints = []

        for hint in hints:
            hint = self._clear_hint(hint)
            if hint == '':
                continue
            cleared_hints.append(hint)

        return cleared_hints
