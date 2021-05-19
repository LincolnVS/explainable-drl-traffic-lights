import numpy as np

class RandomSampler(object):
    
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size
    
    def get_sample(self, buffer):
        return np.random.choice(len(buffer), self.batch_size, replace=True)

class ProportionalSampler(object):
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def uniform_sample(self, buffer):
        r =  np.random.choice(len(buffer), 1, replace=True)[0]
        return r

    def proportional_sample(self, buffer):
        summ = sum([v[2] for v in buffer])
        summ = int(summ)
        if summ <= 0:
            summ = 1
        rand =  np.random.choice(summ, 1, replace=True)[0]
        #print(f"Rand 1 = {rand}")
        res = None
        for i,b in enumerate(buffer):
            if b[2] > rand:
                res = i
                break
            else:
                rand -= b[2]
        if res == None:
            return self.uniform_sample(buffer)
        else:
            return res
    
    def get_sample(self, buffer):
        aux = []
        for _ in range(self.batch_size):
            aux.append(self.proportional_sample(buffer))
        return aux

class ProportionalSampler2(object):
    def __init__(self, buffer_size, batch_size):
        self.batch_size = batch_size
        self.buffer_size = buffer_size

    def uniform_sample(self, buffer):
        r =  np.random.choice(len(buffer), 1, replace=True)[0]
        return r

    def proportional_sample(self, buffer):
        summ = sum([v[2] for v in buffer])
        summ = int(summ)
        if summ <= 0:
            summ = 1
        rand =  np.random.choice(summ, 1, replace=True)[0]
        #print(f"Rand 1 = {rand}")
        res = None
        for i,b in enumerate(buffer):
            if b[2] > rand:
                res = i
                break
            else:
                rand -= b[2]
        if res == None:
            return self.uniform_sample(buffer)
        else:
            return res

    def get_sample(self, buffer):
        aux = []
        #for _ in range(self.batch_size): #O(n)
        #    aux.append(self.proportional_sample(buffer)) #O(m)
        buffer = np.array(buffer) # O(1)
        buffer = buffer[buffer[:,2].argsort()[::-1]] # O(m log(m))
        b_len = len(buffer)  # O(1)
        p = np.arange(2/b_len - 1/(b_len**2),0,-2/(b_len**2)) # O(m)
        choices = np.random.choice(np.arange(b_len),self.batch_size,p=p) # O(m log n)
    
        return choices

