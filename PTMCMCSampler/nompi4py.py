# Dummy class for packages that have no MPI
class MPIDummy(object):
    def __init__(self):
        pass

    def Get_rank(self):
        return 0

    def Get_size(self):
        return 1

    def barrier(self):
        pass

    def send(self, lnlike0, dest=1, tag=55):
        pass

    def recv(self, source=1, tag=55):
        pass

    def Iprobe(self, source=1, tag=55):
        pass

    def scatter(self, sendobj, **kwargs):
        if sendobj is not None:
            return sendobj[0]
        return None

    def bcast(self, obj, **kwargs):
        return obj

    def gather(self, sendobj, **kwargs):
        return [sendobj]


# Global object representing no MPI:
COMM_WORLD = MPIDummy()
