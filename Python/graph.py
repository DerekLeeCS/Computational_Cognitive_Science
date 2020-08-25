import numpy as np
from scipy import sparse

class Graph:
    '''
    Each graph is represented as a Matlab structure with many fields. The most
    important fields in a structure g are described here:

    g.objCount:	        Number of objects in the structure
    g.adjCluster:	    A cluster graph represented as an adjacency matrix 
                        *   A cluster graph is a graph where the nodes 
                        *   correspond to clusters, not individual objects
    g.adjClustersym:    Symmetric version of g.adjcluster
    g.Wcluster:	        A weighted cluster graph. The weight of an edge is the
                        reciprocal of its length
    g.adj:		        A graph including object nodes and cluster nodes. The
                        first g.objcount nodes are object nodes, and the rest
                        are cluster nodes.
    g.W:		        A weighted version of g.adj
    g.z:		        Cluster assignments for the objects
    g.ncomp:	        Number of graph components. Will be 2 for direct
                        products 
                        *   E.g. a cylinder has two components -- a chain and a ring
    g.components:	    Cell array storing the graph components 
    g.extlen:	        Length of all external branches (branches that connect an
                        object to a cluster). Used only if all external branch
                        lengths are tied.
    g.intlen:	        Length of internal branches (branches that connect a
                        cluster to a cluster. Used only if all internal branch 
                        lengths are tied.
    '''
    # Creates an empty graph
    def __init__(self,ps):
        self.type = ps.runps.structname
        self.objCount = ps.runps.nobjects   
        self.sigma = ps.sigmainit
        self.adjCluster = [0]; self.adjClusterSym = [0] 
        self.adj = expand_graph([0], np.arange(1, self.objCount), ps.runps.type)
        self.Wcluster = [0]
        self.W = self.adj
        self.z = np.ones(1, self.objCount)
        self.leafLengths = np.ones(1, self.objCount)
        self.extLen = 1; self.intLen = 1

        if self.type in self.graphSetOuter:
            self.nComp = 1
            self.components = np.full((self.nComp), self.compClass)
            self.components[0].type = self.type


            if self.type in self.graphSetInner1:
                self.components[0].prodCount = 1

            elif self.type == 'tree':
                self.components[0].prodCount = 2

            elif self.type in self.graphSetInner2:
                self.components[0].prodCount = 3
          
        elif self.type == 'grid':
            self.nComp = 2
            self.components = np.full((self.nComp), self.compClass)
            self.components[0].type = 'chain'
            self.components[0].prodCount = 1
            self.components[1].type = 'chain'
            self.components[1].prodCount = 1

        elif self.type == 'cylinder':
            self.nComp = 2
            self.components = np.full((self.nComp), self.compClass)
            self.components[0].type = 'ring'
            self.components[0].prodCount = 1
            self.components[1].type = 'chain'
            self.components[1].prodCount = 1


        for i in range(self.nComp):
            self.components[i].adj = [0];     self.components[i].W = [0]
            self.components[i].adjSym = [0];  self.components[i].Wsym = [0]
            self.components[i].nodeCount = 1; self.components[i].nodeMap = 1        # Should this be array
            self.components[i].edgeCount = 0; self.components[i].edgeMap = [0]
            self.components[i].edgeCountSym = 0; self.components[i].edgeMapSym = [0]
            self.components[i].z = np.ones(1,self.objCount)
            self.components[i].illegal = []
        
        self = combineGraphs(self,ps)
            

    # Used for checking graph type
    graphSetOuter = {
        'partition', 'chain', 'ring', 'tree', 'hierarchy', 'order',
        'dirchain', 'dirring', 'dirhierarchy', 'domtree',
        'connected', 'partitionnoself', 'dirringnoself', 'dirchainnoself',
        'ordernoself', 'dirhierarchynoself', 'dirdomtreenoself',
        'undirchain', 'undirchainnoself', 'undirring', 'undirringnoself',
        'undirhierarchy', 'undirhierarchynoself', 'undirdomtree',
        'undirdomtreenoself', 'connectednoself'
    }
    graphSetInner1 = {
        'partition', 'chain', 'ring', 'order',
        'dirchain','dirring', 'connected', 'partitionnoself',
        'dirchainnoself', 'dirringnoself', 'ordernoself', 
        'undirchain', 'undirchainnoself', 'undirring',
        'undirringnoself','connectednoself'
    }
    graphSetInner2 = {
        'hierarchy','dirhierarchy', 'domtree', 'dirhierarchynoself',
        'dirdomtreenoself', 'undirhierarchy', 'undirhierarchynoself',
	    'undirdomtree', 'undirdomtreenoself'
    }

    # Component Class
    class compClass:
        def __init__(self):
            self.type = None
            self.prodCount = None


# ? Type is unused in source code ?
def expand_graph(adj, zs, type):
    '''
    ADJ: A graph over clusters.
    newADJ: A graph over objects and clusters. Created by hanging objects
            off ADJ according to the cluster assignments in zs
    '''
    # Check if adj is a square matrix
    if (adj.shape)[0] != (adj.shape)[1]:
        # Prints error and shape of matrix
        raise Exception("Error in expand_graph: adj must be square (", (adj.shape)[0], ",", (adj.shape)[1], ")")

    # Check if adj matches zs
    elif (adj.shape)[0] != len(zs):
        raise Exception("Error in expand_graph: adj inconsistent with zs (", (adj.shape)[0], " & ", (adj.shape)[1], ")")

    nClust = (adj.shape)[0]
    objCount = 0
    for i in range(len(zs)):
        objCount += len(zs[i])

    # I believe this is supposed to be square
    newADJ = np.zeros(objCount+nClust,objCount+nClust)

    # Source code has objCount+1, but should be objCount in Python
    # b/c MATLAB starts array indexing at 1
    #   * newadj(objcount+1:objcount+nclust, objcount+1:objcount+nclust)=adj;
    for i in range(objCount, objCount+nClust):
        for j in range(objCount, objCount+nClust):
            newADJ[i][j] = adj

    
    # Hang objects off nodes of adj according to zs (assignments) 
    for i in range(len(zs)):
        newADJ[ objCount+i ][ zs[i] ] = 1

    return (newADJ,objCount)



def combineGraphs(graph, ps, *args):
    '''
    Create direct product of components in graph. 
    1) If given ZONLY flag, adjust class assignments only, or 
    2) also adjust graph.W
        a) if given ORIGGRAPH, COMPIND and IMAP and ps.prodtied == 0 and 
    	graph.ncomp > 1, copy across values from ORIGGRAPH.W (COMPIND and
    	IMAP tell us how to find them: IMAP maps nodes in component
    	COMPIND of GRAPH onto their equivalents in ORIGGRAPH.
        b) else set graph.Wcluster to the graph product of the components.

    % If some data are missing, assume they are marked with -1 in the original
    % graph.z. 
    '''
    origGraph = []; compInd = 0; iMap = []
    zOnly = 0

    # Switch case for optional arguments
    for i in range(0,len(args),2):
        if args[i] == 'origgraph':
            origGraph = args[i+1]
        elif args[i] == 'compind':
            compInd = args[i+1]
        elif args[i] == 'imap':
            iMap = args[i+1]
        elif args[i] == 'zonly':
            zOnly = args[i+1]

    compSizes = np.zeros(graph.nComp)
    for i in (graph.nComp):
        compSizes[i] = (graph.components[i].adj.shape)[0]
        # 1:size(graph.components[u].adj,1))'
        graph.components[i].nodeMap = np.transpose(np.arange(1,compSizes[i]+1))
        graph.components[i].edgeMap = get_edgeMap(graph.components[i].adj)
        graph.components[i].edgeMapSym = get_edgeMap(graph.components[i].adjSym, 'sym', 1)

    graph.compSizes = compSizes
    W = graph.components[0].W
    adj = graph.components[0].adj
    z = graph.components[0].z
    illegal = graph.components[0].illegal

    # 2:graph.nComp
    for i in range(1,graph.nComp):
        na = (W.shape)[0]   # Get row length
        Wb = graph.components[i].W; adjB = graph.components[i].adj
        nb = (Wb.shape)[0]; zb = graph.components[i].z
        Wnew = sparse.kron(np.eye(nb), W); adjNew = sparse.kron(np.eye(nb), adj)   # Computes Kronecker Tensor Product
        z += na*(zb-1)
        illegal += na * np.arange(0,nb)                   ######### this is unused too??

        # Won't this create an exponential number of repeated matrices?
        # Each i causes a loop for j
        # Each j repmats itself
        for j in range(i-1):
            graph.components[j].nodeMap = np.matlib.repmat(graph.components[j].nodeMap,nb,1)
            # I read that I should use scipy.sparse.kron instead of 
            # np.kron b/c numpy cannot handle sparse matrices
            graph.components[j].edgeMap = sparse.kron(np.eye(nb), graph.components[j].edgeMap)  
            graph.components[j].edgeMapSym = sparse.kron(np.eye(nb), graph.components[j].edgeMapSym)

        WnewBunScram = sparse.kron(np.eye(na), Wb); adjNewBunScram = sparse.kron(np.eye(na), adjB)
        # Creates nb by na array of increasing numbers (1 to na*nb)
            # E.g. 1 3
            #      2 4
        # Vectorizes the array
            # E.g. 1 2
            #      3 4
            # E.g. 1 3 2 4
        sInd = np.transpose(np.reshape(np.arange(1,na*nb+1), (nb,na), 'F'))
        sInd = np.reshape(sInd,(na*nb,1),'F')

        #### Currently not implemented in scipy as of 8/11 ####
        # Takes a block matrix defined by (sind,sind) and adds to varNew
        # E.g. mat(2:4,2:4) will give all elements contained in
        # (2,2) to (2,4) to (4,4) to (4,2) inclusive
        adj = adjNew + adjNewBunScram[sInd,sInd]
        W = Wnew + WnewBunScram[sInd,sInd]

        graph.components[i].nodeMap = np.matlib.repmat(graph.components[i].nodeMap, na,1)
        graph.components[i].nodeMap = (graph.components[i].nodeMap)[sInd]
        # Looks like this is updating illegal indices?
        # Setting range from newIllegal to nb(na-1) + newIllegal
            # Off by one error in translation?
            # In source, nb*0:(na-1)+newillegal
            # Should start at newIllegal-1?
        newIllegal = graph.components[i].illegal
        illInd = np.zeros(1,nb*na); illInd[nb * np.arange(na) + newIllegal] = 1
        illegal = np.union1d(newIllegal, np.nonzero(illInd[sInd]))

        # Kronecker product with identity will produce na*m x na*n block matrix
        # Assuming edgeMap is m x n
        # The m x n blocks along the diagonal will be the only nonzero components of matrix
        graph.components[i].edgeMap = sparse.kron(np.eye(na), graph.components[i].edgeMap)
        graph.components[i].edgeMap = graph.components[i].edgeMap[sInd,sInd]
        graph.components[i].edgeMapSym = sparse.kron(np.eye(na), graph.components[i].edgeMapSym)
        graph.components[i].edgeMapSym = graph.components[i].edgeMapSym[sInd,sInd]

    graph.compInds = []
    graph.globInds = np.zeros(compSizes,compSizes)

    # Analysis: Components of each node at highest level
    for i in range(graph.nComp):
        graph.compInds[:,i] = graph.components[i].nodeMap
    
    # Converts row and col indices to a linear index
    # E.g. 2,1 in a 3x3 matrix will be equivalent to 2
    inds = np.ravel_multi_index(compSizes, graph.compInds)

    # Synthesis: Map component nodes to combined node
    # Source was 1:size(adj,1)
    graph.globInds[inds] = np.arange(0,(adj.shape)[0])

    # Finds nonzero indices
    # Source was find(graph.z>=0)
    obsInd = np.nonzero(graph.z)
    graph.z[obsInd] = z[obsInd]
    graph.illegal = illegal
    nObj = len(obsInd)

    if not zOnly: 
        graph.adjCluster = adj
        # Creates a boolean matrix representing whether there is a nonzero element at an index
        graph.adjClusterSym = float(graph.adjCluster or np.transpose(graph.adjCluster))
        graph.Wcluster = W
        doubleWcluster = np.matlib.repmat(0, (graph.Wcluster.shape)[0], (graph.Wcluster.shape)[1])
        doubleWcluster[graph.adjCluster and np.transpose(graph.adjCluster)] = graph.Wcluster[graph.adjCluster and np.transpose(graph.adjCluster)]
        graph.WclusterSym = graph.Wcluster + np.transpose(graph.Wcluster) - doubleWcluster

    if (not zOnly) and (not ps.prodTied) and (graph.nComp > 1) and (origGraph.size == 0):
        # Copy across values from origgraph.Wcluster
        (rInd,cInd) = np.nonzero(graph.adjCluster)
        oldComps = graph.compInds
        oldComps[:,compInd] = iMap[oldComps[:,compInd]]
        oldEdgeRsInd = np.ravel_multi_index(origGraph.compSizes, oldComps[rInd,:])
        oldEdgeRs = origGraph.globInds[oldEdgeRsInd]
        oldEdgeCsInd = np.ravel_multi_index(origGraph.compSizes, oldComps[cInd,:])
        oldEdgeCs = origGraph.globInds[oldEdgeCsInd]
        oldW = origGraph.Wcluster
        oldEdgeLengths = oldW(np.ravel_multi_index(oldW.size, oldEdgeRs, oldEdgeCs))
        if (oldW.shape)[0] == 1:
            oldW = 1
        oldEdgeLengths[np.where(oldEdgeLengths == 0)] = np.median(oldW[np.nonzero(oldW)])
        newW = graph.adjCluster
        newW[np.nonzero(adj)] = oldEdgeLengths
        graph.Wcluster = newW

    fullW = np.zeros( (W.shape)[0] + nObj )
    leafInds = np.ravel_multi_index(fullW.size, nObj+z[obsInd], np.arange(0,nObj))
    fullW[leafInds] = graph.leafLengths[obsInd]
    fullW[np.arange(nObj,len(fullW)), np.arange(nObj,len(fullW))] = graph.Wcluster
    graph.W = fullW
    graph.adj = np.nonzero(fullW)
    graph.adjSym = graph.adj or np.transpose(graph.adj)
    graph.Wsym = fullW
    Wtr = np.transpose(fullW)
    graph.Wsym[np.transpose(graph.adj)] = Wtr[np.transpose(graph.adj)]

    return graph