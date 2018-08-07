class TimeIndex:

    def __init__(self, fs=1):
        self.fs = fs

    def __getitem__(self, n):
        return n / self.fs

    def check_same(self, other):
        if not isinstance(other, TimeIndex):
            raise Exception("One object is a TimeIndex; the other is not.")
        elif self.fs != other.fs:
            raise Exception("The sampling rate of the two TimeIndexes are not the same.")

