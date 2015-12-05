from __future__ import print_function
import sys

def query_yes_no(question):
    """Ask the user a 'yes/no' question.
    Make them answer either yes/no - loop until a valid answer has been given.

    Parameter:
      question - string to be displayed as question.

    Returns:
        Bool - True or False depending on the answer.

    Note:
        Inspired by some code found here:
            http://code.activestate.com/recipes/577097/
    """
    valid = {"yes": True, "y": True,
             "no": False, "n": False}
    prompt = " [y/n] "

    while True:
        print(question + prompt, end='')
        choice = raw_input().lower()
        if choice in valid:
            return valid[choice]
        else:
            print("Please respond with 'yes' or 'no' ")
