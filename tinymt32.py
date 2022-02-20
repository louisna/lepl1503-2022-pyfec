"""
This file is just a python implementation of the tinymt32 library, written in C.
It does not contain all the functions of tinymt32-C, but everything necessary
to generate u32 values.

Author:
    Louis Navarre
"""

MIN_LOOP = 8
PRE_LOOP = 8
TINYMT32_MASK = 0x7FFFFFFF
TINYMT32_SH0 = 1
TINYMT32_SH1 = 10
TINYMT32_SH8 = 8
UINT32_MOD = 1 << 32

class TINYMT32:

    def __init__(self, seed):
        self.mat1 = 0x8f7011ee
        self.mat2 = 0xfc78ff1f
        self.tmat = 0x3793fdff
        self.status = [seed, self.mat1, self.mat2, self.tmat]
        
        for i in range(1, MIN_LOOP):
            self.status[i & 3] ^= (i + (1812433253) * (self.status[(i - 1) & 3] ^ (self.status[(i - 1) & 3] >> 30))) % UINT32_MOD
        
        self.period_certification()
        for i in range(PRE_LOOP):
            self.tinymt32_next_state()
    
    def period_certification(self):
        if self.status[0] & TINYMT32_MASK and self.status[1] == 0 and self.status[2] == 0 and self.status[3] == 0:
            self.status[0] = ord('T')
            self.status[1] = ord('I')
            self.status[2] = ord('N')
            self.status[3] = ord('Y')
        
    def tinymt32_next_state(self, a=1):
        x, y = 0, 0

        y = self.status[3]
        x = ((self.status[0] & 0x7FFFFFFF) ^ self.status[1] ^self.status[2]) % UINT32_MOD
        x ^= ((x << TINYMT32_SH0)) % UINT32_MOD
        y ^= ((y >> TINYMT32_SH0) ^ x) % UINT32_MOD
        self.status[0] = self.status[1]
        self.status[1] = self.status[2]
        self.status[2] = (x ^ (y << TINYMT32_SH1)) % UINT32_MOD
        self.status[3] = y
        self.status[1] ^= (-(int(y & 1)) & self.mat1) % UINT32_MOD
        self.status[2] ^= (-(int(y & 1)) & self.mat2) % UINT32_MOD
    
    def tinymt32_generate_uint32(self):
        self.tinymt32_next_state()
        return self.tinymt32_temper()
    
    def tinymt32_temper(self):
        t0, t1 = 0, 0
        t0 = self.status[3]
        t1 = (self.status[0] + (self.status[2] >> TINYMT32_SH8)) % UINT32_MOD
        t0 ^= t1
        t0 ^= (-(int(t1 & 1)) & self.tmat) % UINT32_MOD
        return t0 % UINT32_MOD
    

if __name__ == "__main__":
    tinymt32 = TINYMT32(42)
    for _ in range(10):
        print(tinymt32.tinymt32_generate_uint32())