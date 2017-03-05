import time

seed = int(time.time())

def get_seed():
    global seed
    seed += 1
    return seed
    
