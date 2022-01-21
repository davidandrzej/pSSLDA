"""
Approximate Distributed (AD) [2] parallel inference for LDA with
topic-in-set knowledge (z-label LDA) [1].

[1] z-label LDA

David Andrzejewski and Xiaojin Zhu
Latent Dirichlet Allocation with Topic-in-Set Knowledge, NAACL-SSLNLP 2009
    
[2] Approximate Distributed Latent Dirichlet Allocation (AD-LDA)

David Newman, Arthur Asuncion, Padhraic Smyth, Max Welling
Distributed Algorithms for Topic Models, JMLR 2009
"""
import multiprocessing as MP
import _pickle as CP # replaces cPickle in python3

import pickle

import numpy as NP
import numpy.random as NPR

cimport numpy as NP
DTYPE = NP.int
ctypedef NP.int_t DTYPE_t
FTYPE = NP.float
ctypedef NP.float_t FTYPE_t

import FastLDA as FLDA

NP.import_array() #prevent segfault

# We're only drawing P values from this interval,
# so does not need to be that large...
RANDSEED_MAX = 100000

def getZ(pindices,allconn,finalz):
    """
    Re-assemble the full z-vector from each Sampler's individual pieces
    """
    for myconn in allconn:
        myconn.send('GETZ')
    for (pidx,myconn) in zip(pindices,allconn):
        finalz[pidx] = pickle.loads(myconn.recv())

def trainsetPerplexity(NP.ndarray[NP.int_t, ndim=1] w,
                       NP.ndarray[NP.int_t, ndim=1] d,
                       NP.ndarray[NP.int_t, ndim=1] z,
                       NP.ndarray[NP.float_t, ndim=2] alpha,
                       NP.ndarray[NP.float_t, ndim=2] beta):
    """
    Calculate in-sample per-word perplexity, useful for:
    -validate parallel versus sequential sampler
    -assess sampler convergence
    """    
    # Get dims
    cdef int N = w.shape[0]
    cdef int D = d.max()+1
    cdef int T = beta.shape[0]
    cdef int W = beta.shape[1]   
    # Get counts    
    cdef NP.ndarray[NP.int_t, ndim=2] nw, nd
    (nw,nd) = [NP.array(val,dtype=NP.int) for val in
               FLDA.countMatrices(w,W,d,D,z,T)]
    # Est phi/ theta
    cdef NP.ndarray[NP.float_t, ndim=2] phi, theta
    (phi,theta) = FLDA.estPhiTheta(nw,nd,alpha,beta)
    # Calc perplexity
    return FLDA.perplexity(w,d,phi,theta);

    
def infer(NP.ndarray[NP.int_t, ndim=1] w,
          NP.ndarray[NP.int_t, ndim=1] d,
          NP.ndarray[NP.float_t, ndim=2] alpha,
          NP.ndarray[NP.float_t, ndim=2] beta,
          int numsamp, int randseed,
          int P=1,
          NP.ndarray[NP.int_t, ndim=1] zinit = None,
          reportname = None, reportinterval = None,
          zlabels = None):
    """
    Do LDA inference via parallelized collapsed Gibbs sampling

    Arguments:
    zinit is optional z-initialization (else use 'online' LDA init)

    If reportname != None
    -every <reportintrval> samples dump full z-sample out to disk     
    -reportname string should contain '%d' which will take sample number

    If zlabels != None, do z-label LDA inference
    """
    # Get some dimensions
    cdef int W = beta.shape[1]
    cdef int D = d.max()+1
    cdef int T = beta.shape[0]
    cdef int N = w.shape[0]
    # Build up online initialization
    cdef NP.ndarray[NP.int_t, ndim=1] z
    if(zinit == None):
        print 'Online z initialization'
        z = NP.array(FLDA.onlineInit(w,d,alpha,beta,randseed),dtype=NP.int)
    else:
        z = zinit.copy()
    # Randomly partition the documents
    print 'Assigning documents to partitions'
    cdef NP.ndarray[NP.int_t, ndim=1] docassign = NPR.randint(0,P,(D,))
    partdocs = []
    cdef int p
    for p in range(P):
        partdocs.append(NP.where(docassign == p)[0])
    assert(all([len(pd) > 0 for pd in partdocs]))
    # Get indices associated with each partition
    # (for each idx (< N) put a 1 in the col for the partition (< P))
    print 'Getting indices associated with each partition'
    cdef NP.ndarray[NP.int_t, ndim=2] idxpart
    idxpart = NP.zeros((N,P),dtype=NP.int) # N x P binary matrix
    cdef int i
    for i in range(N):
        idxpart[i,docassign[d[i]]] = 1 # Index i --> partition p
    pindices = []
    for p in range(P):        
        pindices.append(idxpart[:,p].nonzero()[0])
    # WITHIN each partition we need to renumber documents 0,...,Dp-1
    print 'Create re-numbered doc vectors for each partition'
    cdef NP.ndarray[NP.int_t, ndim=1] pidx    
    renumdocs = []
    for pidx in pindices:
        renumdocs.append(d[pidx])
    cdef int doci, doc
    # Temporary doc map for re-mapping indices
    cdef NP.ndarray[NP.int_t, ndim=1] tmpdocmap
    tmpdocmap = NP.zeros((D,),dtype=NP.int)
    # rdocs = vector of doc values for this partition
    # pdocs = doc indices assigned to this partition
    for (pdocs,rdocs) in zip(partdocs,renumdocs):
        # Construct mapping from old-->new doc indices for this partition
        for (pdoci,pdoc) in enumerate(pdocs):
            tmpdocmap[pdoc] = pdoci
        # Re-label di --> position of di in pdocs
        for doci in range(rdocs.shape[0]):
            rdocs[doci] = tmpdocmap[rdocs[doci]]           
    # Initialize local count matrices
    print 'Initializing count matrices'
    (localnws,localnds) = zip(*[[NP.array(val,dtype=NP.int)
                                 for val in
                                 FLDA.countMatrices(w[pidx],W,rd,rd.max()+1,
                                                  z[pidx],T)]
                                for (pidx,rd) in zip(pindices,renumdocs)])
    # Create Sampler processes
    print 'Launching Sampler processes'
    NPR.seed(randseed)
    (allconn,allsamp) = ([],[])
    for (sampi,(locnw,locnd,pidx,rdocs)) in enumerate(zip(localnws,
                                                          localnds,
                                                          pindices,
                                                          renumdocs)):
        # Connections for duplex communications with Sampler
        (myconn,sampconn) = MP.Pipe()
        allconn.append(myconn)
        # SUBTLE BUG DANGER!!!
        # Sampler processes will be incrementing randseed btwn FLDA calls,
        # therefore we DO NOT want to assign Samplers sequential randseeds
        # (because subsequent calls to Samplers would "overlap")
        srandseed = NPR.randint(0, RANDSEED_MAX)
        # Do we have z-labels?
        if(zlabels != None):
            curzl = [zlabels[i] for i in pidx]
            allsamp.append(Sampler(sampconn,w[pidx],rdocs,z[pidx],alpha,beta,
                                   locnw,locnd,srandseed,curzl))
        else:
            allsamp.append(Sampler(sampconn,w[pidx],rdocs,z[pidx],alpha,beta,
                                   locnw,locnd,srandseed))
        # Launch it
        allsamp[-1].start()
    # Init globalnw
    print 'Computing global nw count matrix'
    cdef NP.ndarray[NP.int_t, ndim=2] globalnw = NP.zeros((W,T),dtype=NP.int)
    for localnw in localnws:
        globalnw += localnw
    # Do samples
    cdef NP.ndarray[NP.int_t, ndim=1] finalz
    finalz = NP.zeros((w.shape[0],),dtype=NP.int)
    cdef int si
    perplex = []
    for si in range(numsamp):
        print 'Sample %d of %d' % (si,numsamp)
        # Send globalnw to each process (will launch inference)
        for (myconn,localnw) in zip(allconn,localnws):
            myconn.send((globalnw - localnw).dumps())
        # Collect results
        localnws = [pickle.loads(myconn.recv()) for myconn in allconn]
        # Re-calculate globalnw
        globalnw = NP.zeros(localnws[0].shape,dtype=NP.int)
        for localnw in localnws:
            globalnw += localnw                             
        # If perplex_interval != None, record trainset perplexity
        if(reportname != None and (NP.mod(si,reportinterval) == 0)):
            # Construct finalz out of parallel samples
            getZ(pindices,allconn,finalz)
            CP.dump(finalz,open(reportname % si,'w'))

    # Construct finalz out of parallel samples
    getZ(pindices,allconn,finalz)
    
    # Shut down Sampler processes
    for myconn in allconn:
        myconn.send('KILL')

    # Return finalz
    return finalz


class Sampler(MP.Process):
    """
    A single parallel Gibbs sampler which works on a subset of the documents
    """

    def __init__(self,sampconn,w,d,z,alpha,beta,
                 localnw,localnd,randseed,zlabels=None):
        # IPC pipe connection
        self.sampconn = sampconn
        # LDA data structures
        (self.w,self.d,self.z) = (w,d,z)
        (self.alpha,self.beta) = (alpha,beta)
        (self.localnw,self.localnd) = (localnw,localnd)
        self.randseed = randseed
        # z-labels (might be None)
        self.zlabels = zlabels
        # Superclass (Process) constructor
        super(Sampler,self).__init__()

    def run(self):
        """
        Each Sampler only takes action in response to comm
        input on its Pipe Connection
        """
        while(True):
            # VERY IMPORTANT TO INC RANDSEED, OTHERWISE ALL
            # CALLS TO parallelGibbs WILL BE 'the same' !!!
            self.randseed += 1
            recval = self.sampconn.recv()
            if(isinstance(recval,str) and recval == 'GETZ'):
                # Request for z-vector
                self.sampconn.send(self.z.dumps())
            elif(isinstance(recval,str) and recval == 'KILL'):
                # We're done!  Shut it down...
                break
            else:
                # Else assume we have been passed globalnd, run inference
                globalnw = pickle.loads(recval)
                if(self.zlabels == None):
                    # Standard LDA
                    FLDA.standardGibbs(self.w,self.d,self.z,
                                       self.alpha,self.beta,
                                       self.localnw,self.localnd,globalnw,
                                       self.randseed)
                else:
                    # z-label LDA
                    FLDA.zLabelGibbs(self.zlabels,self.w,self.d,self.z,
                                     self.alpha,self.beta,
                                     self.localnw,self.localnd,globalnw,
                                     self.randseed)
                # Return local nw count matrix
                self.sampconn.send(self.localnw.dumps())
