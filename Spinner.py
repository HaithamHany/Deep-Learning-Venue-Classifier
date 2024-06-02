import os
import sys
import time
import threading


class Spinner:
    """Spinner class to show a simple spinning cursor on the console."""

    def __init__(self, message="Loading heuristics..."):
        self.busy = False
        self.delay = 0.1  # You can adjust spinner speed here
        self.message = message


    def set_msg(self, msg):
        self.message = msg

    def spinner_task(self):
        spinner_chars = "⣾⣽⣻⢿⡿⣟⣯⣷ ⠁⠂⠄⡀⢀⠠⠐⠈"  # Simple spinner characters
        while self.busy:
            for char in spinner_chars:
                sys.stdout.write(f'\r{self.message} {char}')
                sys.stdout.flush()
                time.sleep(self.delay)

    def start(self):
        self.busy = True
        threading.Thread(target=self.spinner_task).start()

    def stop(self):
        self.busy = False
        time.sleep(self.delay)
        sys.stdout.write('\r' + ' ' * (len(self.message) + 2) + '\r')  # Clear the spinner and message

