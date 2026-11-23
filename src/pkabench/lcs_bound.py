"""Optional exact rejection bound; surviving pairs use the published aligner.

An alignment cannot contain more identical pairs than the longest common
subsequence. Reject only when this exact upper bound is below the required
number of identities. This is independent of affine-gap alignment tie breaks.
Build lcs_bound.cpp as a local shared library on a compute allocation.
"""
import ctypes
import numpy as np


class LCSBound:
    def __init__(self,library,sequences,targets):
        self.lib=ctypes.CDLL(str(library))
        self.lib.ka_lcs_create.argtypes=[ctypes.POINTER(ctypes.c_char_p),ctypes.POINTER(ctypes.c_int),ctypes.POINTER(ctypes.c_int),ctypes.c_int]
        self.lib.ka_lcs_create.restype=ctypes.c_void_p
        self.lib.ka_lcs_bounds.argtypes=[ctypes.c_void_p,ctypes.c_char_p,ctypes.c_int,ctypes.POINTER(ctypes.c_int)]
        self.lib.ka_lcs_bounds.restype=ctypes.c_int
        self.lib.ka_lcs_destroy.argtypes=[ctypes.c_void_p]
        encoded=[sequences[t['sequence_key']].encode('ascii') for t in targets]
        lengths=(ctypes.c_int*len(encoded))(*(len(t) for t in encoded))
        percents=(ctypes.c_int*len(encoded))(*({'PROTEIN':40,'PEPTIDE':80}[t['antigen_type']] for t in targets))
        pointers=(ctypes.c_char_p*len(encoded))(*encoded)
        self.pointer=self.lib.ka_lcs_create(pointers,lengths,percents,len(encoded))
        if not self.pointer:raise MemoryError('Cannot initialise exact LCS bound')
        self.output=np.empty(len(targets),dtype=np.int32)

    def bounds(self,query):
        encoded=query.encode('ascii')
        result=self.lib.ka_lcs_bounds(self.pointer,encoded,len(encoded),self.output.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
        if result:raise RuntimeError('Exact LCS bound failed')
        return self.output

    def close(self):
        if self.pointer:self.lib.ka_lcs_destroy(self.pointer);self.pointer=None

    def __del__(self):
        if getattr(self,'pointer',None):self.close()
