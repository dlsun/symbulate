class TimeIndex:

    def __init__(self, fs=1):
        self.fs = fs

    def __getitem__(self, n):
        return n / self.fs
