import numpy as np

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

class Graph:
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

    class compClass:
        def __init__(self):
            self.type = None
            self.prodCount = None