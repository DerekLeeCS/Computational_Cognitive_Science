# Terms
A form is a high-level type of structure. Examples given in the paper are:

> a tree, a ring, a dimensional order, a set of clusters, or some other kind of configuration

Structures are the instances of a specific form. Searching for the best structure means finding the best instance of the form that represents the data. An example given in the paper is:

> Biologists have long agreed that tree structures are useful for organizing living kinds but continue to debate which tree is best—for instance, are crocodiles better grouped with lizards and snakes or with birds (8)?

# Algorithm

To find the best structure that explains the data ( maximize P(S, F|D) ), search through every form and find the best structure for each form. The final structure is the best structure overall. 

Begin with all entities in a single cluster node and use graph grammars to split 
entities into multiple cluster nodes.
At every split, entities in original cluster node  must be distributed between
the 2 new cluster nodes.
2 entities are randomly chosen and 1 is placed in each of the new cluster nodes.
The rest of the entities are selected randomly and placed greedily in a cluster node.

During each iteration, each cluster node is split multiple times, and the 
split that improves the score the most is accepted.


# masterrun.m

### Source
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

### Source
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

Displays the "real" structure and the best estimated structure.

Calls [setrunps.m](#setrunps.m "Goto setrunps.m")
Calls [scaledata.m](#scaledata.m "Goto scaledata.m")
Calls [draw_dot.m](#draw_dot.m "Goto draw_dot.m")
Calls [relgraphinit.m](#relgraphinit.m "Goto relgraphinit.m")
Calls [structcounts.m](#structcounts.m "Goto structcounts.m")
Calls brlencases (in runmodel.m)

### Source

> % deal with different approaches to branchlengths at current speed
> `function [ll graph, bestglls, bestgraph, ps] = brlencases(data, ps, graph, bestglls, bestgraph, savefile)`

Calls [structurefit.m](#structurefit.m "Goto structurefit.m")

# setrunps.m

### Source
> `function [nobjects, ps]=setrunps(data, dind, ps)`
> 
> % initialize runps component of ps

Appears to set parameters for running the model. 

# scaledata.m

### Source
> `function [data  ps]= scaledata(data, ps)`
> 
> % scale the data according to several strategies

Scales data according to type of data. Can perform no scaling, center the matrix, make mean =  0 and max covariance = 1, and make data look like similarity matrix (max covariance = 1, min covariance ~ 0)

Calls [simpleshift.m](#simpleshift.m "Goto simpleshift.m")
Calls [makesimlike.m](#makesimlike.m "Goto makesimlike.m")
Calls makechunks (in scaledata.m)

### Source
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

### Source
> `function data = simpleshiftscale(data, ps)`
> 
> % shift and scale data so that mean is zero, and largest covariance is
> 1

One of the functions that implements a specific scaling of data.

## makesimlike.m

### Source
> `function data = makesimlike(data, ps)`
> 
> % shift and scale DATA so that the maximum value in covariance is 1,
> and the % smallest value is 0 (not always possible, in which case we
> settle for the % closest value to 0 we can get).

One of the functions that implements a specific scaling of data.

# draw_dot.m

### Source
> `function [xret, yret, labels] = draw_dot(adj, labels, varargin)`
> % 
> % [x, y, labels] = draw_dot(adj, lables)   draw a graph defined by adjacency matrix  
> %   
> % Sample code illustrating use of graph_to_dot and dot_to_graph.m functions 
> %     for interfacing  GraphViz layout and Matlab UI powers   
> % 
> % (C) Dr. Leon Peshkin  pesha @ ai.mit.edu  /~pesha     24 Feb 2004

A set of functions written by Dr. Leon Peshkin to draw a graph.

Calls [graph_to_dot.m](#graph_to_dot.m "Goto graph_to_dot.m")
Calls [dot_to_graph.m](#dot_to_graph.m "Goto dot_to_graph.m")
Calls [my_setdiff.m](#mysetdiff.m "Goto mysetdiff.m")
Calls [graph_draw.m](#graph_draw.m "Goto graph_draw.m")

## graph_to_dot.m

### Source
> `function graph_to_dot(adj, varargin)`
>
> % graph_to_dot(adj, VARARGIN)  Creates a GraphViz (AT&T) format file representing
%                     a graph given by an adjacency matrix.
%  Optional arguments should be passed as name/value pairs [default]
%
%   'filename'  -  if omitted, writes to 'tmp.dot'
%  'arc_label'  -  arc_label{i,j} is a string attached to the i-j arc [""]
% 'node_label'  -  node_label{i} is a string attached to the node i ["i"]
%  'width'      -  width in inches [10]
%  'height'     -  height in inches [10]
%  'leftright'  -  1 means layout left-to-right, 0 means top-to-bottom [0]
%  'directed'   -  1 means use directed arcs, 0 means undirected [1]
%
% For details on dotty, See http://www.research.att.com/sw/tools/graphviz
%
% by Dr. Leon Peshkin, Jan 2004      inspired by Kevin Murphy's  BNT
%    pesha @ ai.mit.edu /~pesha

## dot_to_graph.m

### Source

> `function [Adj, labels, x, y] = dot_to_graph(filename)`
> 
> % [Adj, labels, x, y] = dot_to_graph(filename)
% Extract an adjacency matrix, node labels, and layout (nodes coordinates) 
% from GraphViz file       http://www.research.att.com/sw/tools/graphviz
%
% INPUT:  'filename' - the file in DOT format containing the graph layout.
% OUTPUT: 'Adj' - an adjacency matrix with sequentially numbered edges; 
%     'labels'  - a character array with the names of the nodes of the graph;
%          'x'  - a row vector with the x-coordinates of the nodes in 'filename';
%          'y'  - a row vector with the y-coordinates of the nodes in 'filename'.
%
% NOTEs: not guaranted to parse ANY GraphViz file. Debugged on undirected 
%       sample graphs from GraphViz(Heawood, Petersen, ER, ngk10_4, process). 
%       Complaines about RecursionLimit on huge graphs.
%       Ignores singletons (disjoint nodes). Handles loops (arc to self).          
% Sample DOT code "ABC.dot", read by [Adj, labels, x, y] = dot_to_graph('ABC.dot')
% Plot by    draw_graph(adj>0, labels, zeros(size(x,2),1), x, y);  % from BNT
% digraph G {
%       A [pos="28,31"];
%       B [pos="74,87"];
%       A -- B [pos="e,61,71 41,47 46,53 50,58 55,64"];
% }
%                                                last modified: 24 Feb 2004
% by Dr. Leon Peshkin: pesha @ ai.mit.edu | http://www.ai.mit.edu/~pesha 
%  & Alexi Savov:  asavov @ wustl.edu |  http://artsci.wustl.edu/~azsavov


## mysetdiff.m
Assuming my_setdiff.m refers to mysetdiff.m

### Source

> `function C = mysetdiff(A,B)`
% MYSETDIFF Set difference of two sets of positive integers (much faster than built-in setdiff)
% C = mysetdiff(A,B)
% C = A \ B = { things in A that are not in B }

Simple optimization for a built-in MATLAB function.

## graph_draw.m

### Source

> `function [x, y, h] = graph_draw(adj, varargin)`
%  [x, y, h] = graph_draw(adj, varargin)  
%
% INPUTS:      ADJ   -  Adjacency matrix (source, sink)
%      'linestyle'   -  default '-' 
%      'linewidth'   -  default .5
%      'linecolor'   -  default Black
%      'fontsize'    -  fontsize for labels, default 8 
%      'node_labels' -  Cell array containing labels <Default : '1':'N'>
%      'node_shapes' -  1 if node is a box, 0 if oval <Default : zeros>
%      'X'  Coordinates of nodes on the unit square <Default : calls make_layout>
%      'Y'     
%
% OUTPUT:   x, y   -  Coordinates of nodes on the unit square
%               h  -  Object handles [h(i,1) is the text handle - color
%                                     h(i,2) is the circle handle - facecolor]
% NOTES: 
%          Shades  nodes linked to self ! 
>
> % 24 Feb 2004  cleaned up, optimized and corrected by Leon Peshkin pesha @ ai.mit.edu 
% Apr-2000  draw_graph   Ali Taylan Cemgil   <cemgil@mbfys.kun.nl> 
% 1995-1997 arrow        Erik A. Johnson     <johnsone@uiuc.edu>

Draws a graph based on input parameters.

# relgraphinit.m

### Source
> `function graph = relgraphinit(data, z, ps)`
> 
> % Create initial graph for relational data set DATA by using various %
> heuristics

Calls [makelcfreq.m](#makelcfreq.m "Goto makelcfreq.m")
Calls [makeemptygraph.m](#makeemptygraph.m "Goto makeemptygraph.m")
Calls chooseinithead (in relgraphinit.m)
Calls growgraph (in relgraphinit.m)
Calls finishgraph (in relgraphinit.m)
Calls [get_edgemap.m](#get_edgemap.m "Goto get_edgemap.m")
Calls [combinegraphs.m](#combinegraphs.m "Goto combinegraphs.m")
Unsure of reason for calling combinegraphs.m.


### Source
> `function [head, tail, used] = chooseinithead(lc, lcprop, graph)`

Has an `if 0` statement. Unclear of purpose (should never be true).
Chooses a head and a tail based on the type of structure. 

### Source
> `function [graph, head, tail, used] = growgraph(graph, head, tail, used, lc, lcprop)`

Inserts a node in the graph depending on the type of structure. Currently unimplemented for tree structures.
Appears that `used` is a bool vector representing whether or not the corresponding element has been placed in the graph.

### Source
> `function graph = finishgraph(graph, head, tail)`

Only applies for rings. If the tail and the head are not connected, connects them.

## makelcfreq.m

> `function lc = makelcfreq(R, zs)`

Unsure what this does. Google search brings references to LC circuits (probably wrong). 

>     nclass= length(unique(zs));
>     lc=zeros(nclass);
>     [r, c]=ind2sub(size(R), find(R));
> 
>     for i = 1:length(r)
>       lc(zs(r(i)), zs(c(i))) = lc(zs(r(i)), zs(c(i))) + R(r(i), c(i));
>     end

Appears to loop through diagonal of R and zs. lc has something to do with number of unique clusters. Maybe likelihood? Not sure of how that would make sense.

## makeemptygraph.m

### Source
> `function graph  = makeemptygraph(ps)`
> 
> % Make a graph with one cluster and no objects.

Initializes the graph class. Does not contain any data. Initialization depends on type of structure.
Calls [combinegraphs.m](#combinegraphs.m "Goto combinegraphs.m")
Unsure of reason for calling combinegraphs.m.

## get_edgemap.m

### Source

> `function emap = get_edgemap(adj, varargin)`
> 
> % Create an edge map (EMAP) which associates each edge in ADJ with a %
> number

Finds number of non-zero values in adjacency matrix and creates a vector from 1 to number of non-zero values. 

## combinegraphs.m 

### Source
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


# structcounts.m

### Source
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

Calls [gridpriors.m](#gridpriors.m "Goto gridpriors.m")

## gridpriors.m

### Source
> `function lps = gridpriors(maxn, theta, T, type)`
> 
> % Count number of ways to put objects on a grid or cylinder.
> 
> % THETA: parameter for geometric distribution on cluster number %
> T(n,k): number of ways to put n elements into k parcels

Used to calculate priors for grids and cylinders.

# structurefit.m

### Source

> `function [ll, graph, bestgraphlls, bestgraph] = structurefit(data, ps, graph, savefile)` 
> 
> % Fit a given structure to matrix DATA using parameters PS 
> 
> %  Graph representation: 
> %   graph.adj: adjacency matrix 
> %   graph.objcount: number of object nodes (as distinct from cluster nodes) 
> %   graph.W: edge weights (w = 1/distance)

Starts with a single cluster node (makeemptygraph.m). Greedily splits the cluster nodes as long as the score keeps improving. Uses heuristics to split. 

Calls [makeemptygraph.m](#makeemptygraph.m "Goto makeemptygraph.m")
Calls optimizebranches (in structurefit.m)
Calls graphscorenoopt (in structurefit.m)
Calls [choose_node_split.m](#choose_node_split.m "Goto choose_node_split.m")
Calls optimizedepth.m (in structurefit.m)


### Source
> `function [ll graph]= optimizebranches(graph, data, ps)`

Computes log likelihood of a structure. Calculates log p(DATA|GRAPH) \*  log p(GRAPH), which is equivalent to log( p(DATA|GRAPH) \* p(GRAPH) ), Bayes' Theorem. 

Calls [graph_like.m](#graph_like.m "Goto graph_like.m")
Calls [graph_prior.m](#graph_like.m "Goto graph_prior.m")

### Source

> % approximate graph score (no integrating out) 
> `function [ll graph]= graphscorenoopt(graph, data, ps)`

Unsure what this does. The only difference between this function and optimizebranches is this function has `ps.fast = 1`, whereas optimizebranches has `ps.fast = 0`. Maybe there was an error in the code:

    [currprob graph]= optimizebranches(graph, data, ps);
    if ps.speed == 5
      [currprob graph]= graphscorenoopt(graph, data, ps); 
    end

Maybe this was supposed to be if/else. That would mean that this function sets a flag to approximate the scores instead of making exact calculations.

### Source

> `function [lls newgraph]= optimizedepth(graph, depth, lls,  newgraph, data, ps)`  
> % optimize all splits at DEPTH

Only splits cluster nodes. Does not split object nodes.

Calls optimizebranches (in structurefit.m)

## graph_like.m

### Source

> `function [logI graph] = graph_like(data, graph, ps)`
> 
> % graph_like(data, adj, beta, sigma): compute log p(DATA|GRAPH)

Uses one of two functions to compute log likelihood, depending on if the data is a relational data set or if it has feature or similarity data. 

Calls [graph_like_rel.m](#graph_like_rel.m "Goto graph_like_rel.m")
Calls [graph_like_conn.m](#graph_like_conn.m "Goto graph_like_conn.m")


## graph_like_rel.m

### Source
> `function [logI graph] = graph_like_rel(data, graph, ps)`
> 
> % graph_like_rel(data, adj, beta, sigma): compute log p(DATA|GRAPH), where 
> %   D is a relational data set

One of the functions that computes log likelihood of the data given the structure.

## graph_like_conn

### Source
> `function [logI graph] = graph_like_conn(data, graph, ps)`
> 
> % Compute log P(DATA|GRAPH) 
> % D: feature data or similarity data

One of the functions that computes log likelihood of the data given the structure.

## graph_prior.m

### Source
> `function gp = graph_prior(graph, ps)`
> 
> % Compute prior on graph GRAPH.
> 
> % GP = log P(GRAPH). 
> 
> % if some objects are out of play, gp will be proportional to the true
> prior % but the actual number will be wrong

Uses the probabilities computed in [structcounts.m](#structcounts.m "Goto structcounts.m")

## choose_node_split.m

### Source 

> `function [ll, part1, part2, newgraph]=choose_node_split(graph, compind, splitind, pind, data, ps)`
> 
> % split node SPLITIND in component COMPIND using production PIND

This is the part of the algorithm where a cluster node is split into two cluster nodes. 2 entities are randomly chosen 

Calls [choose_seedpairs.m](#choose_seedpairs.m "Goto choose_seedpairs.m")
Calls [best_split.m](#best_split.m "Goto best_split.m")

## choose_seedpairs.m

### Source
> `function seedpairs = choose_seedpairs(graph, compind,c,pind,  ps)`
> 
> % Choose pairs to seed split of node C in component PIND.
> 
> % C:	   cluster node to split 
> % COMPIND: current graph component (-1 for top level split) 
> % PIND : production to try

Returns a matrix of pairs of entities to use as the initial split.

## best_split.m

### Source

> `function [ll, part1, part2, newgraph] = best_split(graph, compind, c, pind, data, seedpairs, ps)`
> 
> % Choose the best split of cluster node C.
> 
> % SEEDPAIRS: objects to seed the new children nodes 
> % COMPIND:   graph component (-1 for high level split) 
> % PIND :     which grammar to use

Splits the cluster node multiple times, and chooses the split that results in the best score.
At first, randomly assigns 1 entity to each child, and then greedily assigns the rest. 

Calls [split_node.m](#best_split.m "Goto split_node.m")
Calls [empty_graph.m](#best_split.m "Goto empty_graph.m")
Calls [add_element.m](#add_element.m "Goto add_element.m")
Calls [graph_like.m](#graph_like.m "Goto graph_like.m")
Calls [graph_prior.m](#graph_like.m "Goto graph_prior.m")
Calls [draw_dot.m](#draw_dot.m "Goto draw_dot.m")
Calls [simplify_graph.m](#simplify_graph.m "Goto simplify_graph.m")

## split_node.m

### Source
> `function [graph, c1, c2] = split_node(graph, compind, c, pind, part1, part2, ps)`
> 
> % split node C in component CIND using production PIND and put PART1 and PART2 
> % in the two children

Splits a node using graph grammars and puts PART1 and PART2 in the children. Not sure what PART1 and PART2 appear to be entities, but this doesn't make sense, because entities are assigned in add_element.m.

Calls [combinegraphs.m](#combinegraphs.m "Goto combinegraphs.m")
Unsure of reason for calling combinegraphs.m.

## empty_graph.m

### Source

> `function graph = empty_graph(graph, compind, c1, c2)`
> 
> % Remove all members of cluster C1 or cluster C2 from component COMPIND of 
> % GRAPH

Looks like this is used after split_node.m. One PART1 and PART2 were placed in the child nodes, empty_graph.m is called to remove PART1 and PART2 in the original cluster.

## add_element.m

### Source

> `function g = add_element(g, compind, c, element, ps)`
> 
> % add entity ELEMENT to cluster C1 of component COMPIND

Used to split nodes. 

## simplify_graph.m

### Source

> `function graph = simplify_graph(graph, ps)`
> 
> % Try cleaning up GRAPH by removing unnecessary cluster nodes.
> 
> % remove  cluster nodes we don't want 
> %	1) dangling cluster nodes (any node that's not an object node, but has 
> %				    exactly one (or zero) cluster neighbors and 
> %				    no object neighbors) 
> %	2) any cluster node with exactly two neighbors, one of which is a 
> %	    cluster node

This is necessary because the algorithm used sometimes makes unnecessary splits. 

Calls redundantinds (in simplify_graph.m)

### Source

> `function [adj W z includeind]= redundantinds(caseind, graph, i, adj, W, z, occ, ps)`
> 
> % XXX: should really apply these only if we haven't tied branch lengths. But 
> % since we only tie branches as a search heuristic right now, it doesn't matter 
> % too much.

Calculates indices of redundant cluster nodes.

## gibbs_clean.m

### Source

> `function [ll graph] = gibbs_clean(graph, data, ps, varargin)`
> 
> % SWAPTYPES: which swaps to include 
> % 1. individual objects 
> % 2. cluster nodes 
> % 3. subtreeprune (objects and cluster nodes) 
> % 4. at level of entire graph 
> % 5. remove dimensions

### Notes

>     % try swapping objects before removing clusters. This gives a new cluster
>     % the chance to establish itself. Mainly added (5/8/06) to allow the
>     % algorithm to find the best ring for the Kula data. I haven't tested this
>     % for feature/similarity data but suspect it might not be good for two
>     % reasons: 
>     %	a) for feature data, we initially tie branches together and therefore
>     %	encourage the model to introduce more clusters than is correct. We
>     %	might not want to make it easier for extra clusters to stick around. 
>     %	b) it's probably too expensive on the bigger data sets.

Attempts to improve score by swapping objects within the graph. 
Uses Hessian to calculate score and optimizes based on that.