# Algorithm

Begin with all entities in a single cluster node and use graph grammars to split 
entities into multiple cluster nodes.
At every split, entities in original cluster node  must be distributed between
the 2 new cluster nodes.
2 entities are randomly chosen and 1 is placed in each of the new cluster nodes.
The rest of the entities are selected randomly and placed greedily in a cluster node.

During each iteration, each cluster node is split multiple times, and the 
split that improves the score the most is accepted.


# masterrun.m

> ## Source
>The top level file is masterrun.m. Type 
>`>> masterrun`
> and the code will fit three forms (chain, ring and tree) to three data
> sets. If 'neato' is installed (part of the GraphViz package -- see
> below) then one figure window will show the progress of the current
> search, and a second will show the best solution found for the data
> set most recently analyzed.
> 
> To run other analyses, edit masterrun.m to specify the structures and
> data sets you want to explore, and the number of times to run each
> analysis.

Calls [runmodel.m](#runmodel.m "Goto runmodel.m")

# runmodel.m

> ## Source
> `function [ll, graph, names, bestglls, bestgraph] =  runmodel(ps, sind, dind, rind, savefile)`
> 
> ## Given data set DIND, find the best instance of form SIND.  
> %   PS:	 parameter structure with hyperparameters, etc 
> %   RIND:     which repeat is this? 
> %   SAVEFILE: where to save interim results 
> ## Output:
> %   LL:	       log probability of the best structure found 
> %   GRAPH: the best structure found 
> %   BESTGLLS:  log probabilities of the structures explored along the way 
> %   BESTGRAPH: structures exploredalong the way

Calls [setrunps.m](#setrunps.m "Goto setrunps.m")
Calls [scaledata.m](#scaledata.m "Goto scaledata.m")
Calls [relgraphinit.m](#relgraphinit.m "Goto relgraphinit.m")
Calls [structcounts.m](#structcounts.m "Goto structcounts.m")

## setrunps.m

> `function [nobjects, ps]=setrunps(data, dind, ps)`
> 
> % initialize runps component of ps

Appears to set parameters for running the model. 

# scaledata.m

> `function [data  ps]= scaledata(data, ps)`
> 
> % scale the data according to several strategies

Scales data according to type of data. Can perform no scaling, center the matrix, make mean =  0 and max covariance = 1, and make data look like similarity matrix (max covariance = 1, min covariance ~ 0)

Calls [simpleshift.m](#simpleshift.m "Goto simpleshift.m")
Calls [makesimlike.m](#makesimlike.m "Goto makesimlike.m")
Calls makechunks (in scaledata.m)

> `function ps = makechunks(data, ps)`

"Chunk" appears to be a term from cognitive psychology. Essentially a class. If correct, function organizes data into classes.

Example of a chunk:

    friend f34 {
      name Joan
    }
    friend {
      name Jenny
      likes f34
    }
   
## simpleshift.m

> `function data = simpleshiftscale(data, ps)`
> 
> % shift and scale data so that mean is zero, and largest covariance is
> 1

One of the functions that implements a specific scaling of data.

## makesimlike.m

> `function data = makesimlike(data, ps)`
> 
> % shift and scale DATA so that the maximum value in covariance is 1,
> and the % smallest value is 0 (not always possible, in which case we
> settle for the % closest value to 0 we can get).

One of the functions that implements a specific scaling of data.

# relgraphinit.m

> ## Source
> `function graph = relgraphinit(data, z, ps)`
> 
> % Create initial graph for relational data set DATA by using various %
> heuristics

Calls [makeemptygraph.m](#makeemptygraph.m "Goto makeemptygraph.m")
Calls [split_node.m](#split_node.m "Goto split_node.m")

## makeemptygraph.m


## split_node.m

> ## Source
> `function [graph, c1, c2] = split_node(graph, compind, c, pind, part1, part2, ps)`
> 
> % split node C in component CIND using production PIND and put PART1 and PART2 
> % in the two children

Splits a node using graph grammars based on the type of the component. 

Calls combinegraphs.m.
Unsure what the purpose is.

## combinegraphs.m 

> ## Source
> `function graph = combinegraphs(graph, ps, varargin)`
> 
> % create direct product of components in graph.  
> ## 1) If given ZONLY flag, adjust class assignments only, or  
> ## 2) also adjust graph.W 
> a) if given ORIGGRAPH, COMPIND and IMAP and ps.prodtied == 0 and 
> %	graph.ncomp > 1, copy across values from ORIGGRAPH.W (COMPIND and
> %	IMAP tell us how to find them: IMAP maps nodes in component
> %	COMPIND of GRAPH onto their equivalents in ORIGGRAPH.
> 
> b) else set graph.Wcluster to the graph product of the components.
> 
> % If some data are missing, assume they are marked with -1 in the
> original % graph.z.

Appears to have something to do with tensor product for graphs. Given 2 graphs, creates a new graph composed of all combinations of each graph's vertices. E.g. (A,B,C) x (1,2,3) = (A1,A2,A3; B1,... C3)
Unclear why they used the tensor product with the identity matrix. Would only create block matrices along the diagonal.

## structcounts.m

> ## Source
> `function ps = structcounts(nobjects, ps)`
> 
> % make PS.LOGPS: ps.logps{i}(n) is prior for an i-structure with n clusters  
> % We compute these priors in advance and cache them.

Calculates priors for each structure. i represents a specific structure. 

 1. partition, connected
 2. chain
 3. ring
 4. unrooted tree
 5. hierarchy unrooted
 6. rooted hierarchy 
 7. dirchain
 8. dirring
 9. grid
 10. cylinder 

Structures 1-8 are calculated in this file. Structures 9 & 10 are calculated in gridpriors.m.

> ## Bug? 
> Stirling2(n,n) does not result in 1. Only gives 1 when n=1. When n>1, gives an n-by-n matrix. 
> Should give 1 by definition. 
> Looks like result may be from Stirling2(1...n, 1...n)
> 
> %    S2(N,M) represents the number of distinct partitions of N elements 
> %    into M nonempty sets.   
> `s2 = stirling2(maxn,maxn);`

Calls gridpriors.m 

# gridpriors.m

> `function lps = gridpriors(maxn, theta, T, type)`
> 
> % Count number of ways to put objects on a grid or cylinder.
> 
> % THETA: parameter for geometric distribution on cluster number %
> T(n,k): number of ways to put n elements into k parcels

Used to calculate priors for grids and cylinders.