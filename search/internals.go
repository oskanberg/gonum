package search

import (
	"container/heap"
	"math"

	gr "github.com/gonum/graph"
	"github.com/gonum/graph/concrete"
)

type searchFuncs struct {
	successors, predecessors, neighbors    func(gr.Node) []gr.Node
	isSuccessor, isPredecessor, isNeighbor func(gr.Node, gr.Node) bool
	cost                                   gr.CostFunc
	heuristicCost                          gr.HeuristicCostFunc
}

func genIsSuccessor(graph gr.DirectedGraph) func(gr.Node, gr.Node) bool {
	return func(node, succ gr.Node) bool {
		return graph.EdgeTo(node, succ) != nil
	}
}

func genIsPredecessor(graph gr.DirectedGraph) func(gr.Node, gr.Node) bool {
	return func(node, succ gr.Node) bool {
		return graph.EdgeTo(succ, node) != nil
	}
}

func genIsNeighbor(graph gr.Graph) func(gr.Node, gr.Node) bool {
	return func(node, succ gr.Node) bool {
		return graph.EdgeBetween(succ, node) != nil
	}
}

// Sets up the cost functions and successor functions so I don't have to do a type switch every
// time. This almost always does more work than is necessary, but since it's only executed once
// per function, and graph functions are rather costly, the "extra work" should be negligible.
func setupFuncs(graph gr.Graph, cost gr.CostFunc, heuristicCost gr.HeuristicCostFunc) searchFuncs {

	sf := searchFuncs{}

	switch g := graph.(type) {
	case gr.DirectedGraph:
		sf.successors = g.Successors
		sf.predecessors = g.Predecessors
		sf.neighbors = g.Neighbors
		sf.isSuccessor = genIsSuccessor(g)
		sf.isPredecessor = genIsPredecessor(g)
		sf.isNeighbor = genIsNeighbor(g)
	default:
		sf.successors = g.Neighbors
		sf.predecessors = g.Neighbors
		sf.neighbors = g.Neighbors
		sf.isSuccessor = genIsNeighbor(g)
		sf.isPredecessor = genIsNeighbor(g)
		sf.isNeighbor = genIsNeighbor(g)
	}

	if heuristicCost != nil {
		sf.heuristicCost = heuristicCost
	} else {
		if g, ok := graph.(gr.HeuristicCoster); ok {
			sf.heuristicCost = g.HeuristicCost
		} else {
			sf.heuristicCost = NullHeuristic
		}
	}

	if cost != nil {
		sf.cost = cost
	} else {
		if g, ok := graph.(gr.Coster); ok {
			sf.cost = g.Cost
		} else {
			sf.cost = UniformCost
		}
	}

	return sf
}

func NullHeuristic(node1, node2 gr.Node) float64 {
	return 0.0
}

func UniformCost(e gr.Edge) float64 {
	if e == nil {
		return math.Inf(1)
	}

	return 1.0
}

/** Sorts a list of edges by weight, agnostic to repeated edges as well as direction **/

type edgeSorter []concrete.WeightedEdge

func (el edgeSorter) Len() int {
	return len(el)
}

func (el edgeSorter) Less(i, j int) bool {
	return el[i].Cost < el[j].Cost
}

func (el edgeSorter) Swap(i, j int) {
	el[i], el[j] = el[j], el[i]
}

/** Keeps track of a node's scores so they can be used in a priority queue for A* **/

type internalNode struct {
	gr.Node
	gscore, fscore float64
}

/* A* stuff */
type aStarPriorityQueue struct {
	indexList map[int]int
	nodes     []internalNode
}

func (pq *aStarPriorityQueue) Less(i, j int) bool {
	// As the heap documentation says, a priority queue is listed if the actual values
	// are treated as if they were negative
	return pq.nodes[i].fscore < pq.nodes[j].fscore
}

func (pq *aStarPriorityQueue) Swap(i, j int) {
	pq.indexList[pq.nodes[i].ID()] = j
	pq.indexList[pq.nodes[j].ID()] = i

	pq.nodes[i], pq.nodes[j] = pq.nodes[j], pq.nodes[i]
}

func (pq *aStarPriorityQueue) Len() int {
	return len(pq.nodes)
}

func (pq *aStarPriorityQueue) Push(x interface{}) {
	node := x.(internalNode)
	pq.nodes = append(pq.nodes, node)
	pq.indexList[node.ID()] = len(pq.nodes) - 1
}

func (pq *aStarPriorityQueue) Pop() interface{} {
	x := pq.nodes[len(pq.nodes)-1]
	pq.nodes = pq.nodes[:len(pq.nodes)-1]
	delete(pq.indexList, x.ID())

	return x
}

func (pq *aStarPriorityQueue) Fix(id int, newGScore, newFScore float64) {
	if i, ok := pq.indexList[id]; ok {
		pq.nodes[i].gscore = newGScore
		pq.nodes[i].fscore = newFScore
		heap.Fix(pq, i)
	}
}

func (pq *aStarPriorityQueue) Find(id int) (internalNode, bool) {
	loc, ok := pq.indexList[id]
	if ok {
		return pq.nodes[loc], true
	} else {
		return internalNode{}, false
	}

}

func (pq *aStarPriorityQueue) Exists(id int) bool {
	_, ok := pq.indexList[id]
	return ok
}

type denseNodeSorter []gr.Node

func (dns denseNodeSorter) Less(i, j int) bool {
	return dns[i].ID() < dns[j].ID()
}

func (dns denseNodeSorter) Swap(i, j int) {
	dns[i], dns[j] = dns[j], dns[i]
}

func (dns denseNodeSorter) Len() int {
	return len(dns)
}

// General utility funcs

// Rebuilds a path backwards from the goal.
func rebuildPath(predecessors map[int]gr.Node, goal gr.Node) []gr.Node {
	if n, ok := goal.(internalNode); ok {
		goal = n.Node
	}
	path := []gr.Node{goal}
	curr := goal
	for prev, ok := predecessors[curr.ID()]; ok; prev, ok = predecessors[curr.ID()] {
		if n, ok := prev.(internalNode); ok {
			prev = n.Node
		}
		path = append(path, prev)
		curr = prev
	}

	// Reverse the path since it was built backwards
	for i, j := 0, len(path)-1; i < j; i, j = i+1, j-1 {
		path[i], path[j] = path[j], path[i]
	}

	return path
}
