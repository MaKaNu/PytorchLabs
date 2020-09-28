import logging

class CustomFormatter(logging.Formatter):
    """Logging Formatter to add colors and count warning / errors"""

    blue = "\x1b[34;21m"
    green = "\x1b[32;21m"
    yellow = "\x1b[33;21m"
    red = "\x1b[31;21m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: green + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)

    @staticmethod
    def longmsg(msg):
        """tries to slice a str after step elements, if it is a space_element.
        if it is not a space element the str will be sliced at the last
        space_element before the step elements

        Args:
            msg (str): Message as a string.

        Returns:
            list with all the elements with len < step
        """
        assert isinstance(msg, str)
        step = 80
        final_msg = []
        last = 0
        while last + step <= len(msg):
            if msg[last + step] == ' ':
                if msg[last: last + step][0] == ' ':
                    final_msg.append(msg[last + 1: last + step] + '\\n')
                else:
                    final_msg.append(msg[last: last + step] + '\\n')
                last = last + step
            else:
                new_last = msg[last: last + step].rindex(' ')
                if msg[last: last + new_last][0] == ' ':
                    final_msg.append(msg[last + 1: last + new_last] + '\\n')
                else:
                    final_msg.append(msg[last: last + new_last] + '\\n')
                last = last + new_last
                print(last)
        final_msg.append(msg[last + 1:])
        return final_msg


if __name__ == "__main__":
    msg = 'This is a very long message with more than 80 chars to test the method longmsgms from CustomFormatter. This should split the msg in different parts with the len klenght of 80'
    CustomFormatter.longmsg(msg)