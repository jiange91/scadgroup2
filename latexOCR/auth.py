# This is a dummy authentication
import time

def auth(*, userID, imgPath):
    time.sleep(1)
    privilege = 2 * userID
    if privilege < 10:
        print('Not enough credits')
        return (0, {'imgPath': None, 'userID': userID})
    else:
        print('Proceeding')
        return (1, {'imgPath': imgPath, 'userID': userID})