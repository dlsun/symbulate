import time

seed = int(time.time())

def get_seed():
    global seed
    seed += 1
    # ensure that seed does not exceed maximum allowed by Numpy
    if seed > 4e9: seed = 0
    return seed
    
