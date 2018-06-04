// Game generation RPC client in C++.

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <cmath>
#include <cassert>
#include <rpc/client.h>
#include "movegen.hpp"
#include "makemove.hpp"

using std::shared_ptr;
using std::cout;
using std::endl;

constexpr double exploration_parameter = 1.0;

rpc::client* global_rpc_connection;

// Extend Move with a hash.
namespace std {
	template<> struct hash<Move> {
		size_t operator()(const Move& m) const {
			return m.from + m.to * 49;
		}
	};
}

template<int s1, int s2, int s3>
int stride_index(int x, int y, int z) {
	assert(0 <= x and x < s1);
	assert(0 <= y and y < s2);
	assert(0 <= z and z < s3);
	return
		s2 * s3 * x +
		     s3 * y +
			      z;
}

struct pair_hash {
public:
	template <typename T, typename U>
	size_t operator()(const std::pair<T, U>& x) const {
		return std::hash<T>()(x.first) + std::hash<U>()(x.second) * 7;
	}
};

// Map that maps an (x, y) coordinate delta into a layer index in an encoded posterior.
std::unordered_map<std::pair<int, int>, int, pair_hash> position_delta_layers {
	{{-2, -2},  0}, {{-2, -1},  1},
	{{-2,  0},  2}, {{-2,  1},  3},
	{{-2,  2},  4}, {{-1, -2},  5},
	{{-1,  2},  6}, {{ 0, -2},  7},
	{{ 0,  2},  8}, {{ 1, -2},  9},
	{{ 1,  2}, 10}, {{ 2, -2}, 11},
	{{ 2, -1}, 12}, {{ 2,  0}, 13},
	{{ 2,  1}, 14}, {{ 2,  2}, 15},
};

struct Evaluations {
	bool game_over;
	double value;
	std::unordered_map<Move, double> posterior;

	void populate(const Position& board) {
		game_over = false;

		// Build a features map, initialized to all zeros.
		uint8_t feature_buffer[7 * 7 * 4] = {};
		for (int y = 0; y < 7; y++) {
			for (int x = 0; x < 7; x++) {
				// Fill in layer 0 with all ones.
				feature_buffer[stride_index<7, 7, 4>(x, y, 0)] = 1;

				int square = x + 7 * (6 - y);
				uint64_t mask = 1ull << square;
				bool cross_present   = board.pieces[PIECE::CROSS]  & mask;
				bool nought_present  = board.pieces[PIECE::NOUGHT] & mask;
				bool blocker_present = board.blockers              & mask;

				bool piece_present = cross_present or nought_present;
				bool current_player_piece =
					(board.turn == SIDE::CROSS and cross_present) or
					(board.turn == SIDE::NOUGHT and nought_present);
				// If there is a piece of the (player to move)'s then write a 1 to layer 1, otherwise to layer 2.
				if (piece_present) {
					if (current_player_piece)
						feature_buffer[stride_index<7, 7, 4>(x, y, 1)] = 1;
					else
						feature_buffer[stride_index<7, 7, 4>(x, y, 2)] = 1;
				}
				// If there's a blocker write to layer 3.
				if (blocker_present)
					feature_buffer[stride_index<7, 7, 4>(x, y, 3)] = 1;
			}
		}

//		for (int i = 0; i < 7 * 7 * 4; i++)
//			cout << (int)feature_buffer[i];
//		cout << endl;

		// Call the RPC CNN evaluator.
		std::string feature_string(reinterpret_cast<char*>(feature_buffer), sizeof(feature_buffer));
		auto result = global_rpc_connection->call("network", feature_string).as<std::tuple<std::string, double>>();
		std::string& posterior_str = std::get<0>(result);
		value = std::get<1>(result);

		std::ofstream out("/tmp/posterior.bin");
		out << posterior_str;
		out.close();

		const float* posterior_array = reinterpret_cast<const float*>(posterior_str.c_str());
		double softmaxed[7 * 7 * 17];

		// Softmax the posterior array.
		for (int i = 0; i < (7 * 7 * 17); i++)
			softmaxed[i] = exp(posterior_array[i]);
		double total = 0.0;
		for (int i = 0; i < (7 * 7 * 17); i++)
			total += softmaxed[i];
		for (int i = 0; i < (7 * 7 * 17); i++)
			softmaxed[i] /= total;

		// Evaluate movegen.
		Move moves_buffer[2401];
		int num_moves = movegen(board, moves_buffer);
		double total_probability = 0.0;
		for (int i = 0; i < num_moves; i++) {
			Move& move = moves_buffer[i];
			int from_x = move.from % 7;
			int from_y = move.from / 7;
			int to_x   = move.to   % 7;
			int to_y   = move.to   / 7;
			from_y = 6 - from_y;
			to_y   = 6 - to_y;
			double probability;
			if (is_single(move)) {
				probability = softmaxed[stride_index<7, 7, 17>(to_x, to_y, 16)];
			} else {
				std::pair<int, int> delta{to_x - from_x, to_y - from_y};
				int layer_index = position_delta_layers.at(delta);
				probability = softmaxed[stride_index<7, 7, 17>(to_x, to_y, layer_index)];
			}
			posterior.insert({move, probability});
			total_probability += probability;
		}
		// Add a small constant denominator term to prevent diving by zero.
		total_probability += 1e-6;
		// Normalize the posterior.
		for (auto& p : posterior)
			p.second /= total_probability;
	}
};

struct MCTSEdge;
struct MCTSNode;

struct MCTSEdge {
	Move edge_move;
	shared_ptr<MCTSNode> parent_node;
	shared_ptr<MCTSNode> child_node;
	double edge_visits = 0;
	double edge_total_score = 0;

	MCTSEdge(Move edge_move, shared_ptr<MCTSNode> parent_node, shared_ptr<MCTSNode> child_node)
		: edge_move(edge_move), parent_node(parent_node), child_node(child_node) {}

	double get_edge_score() const {
		if (edge_visits == 0)
			return 0;
		return edge_total_score / edge_visits;
	}

	void adjust_score(double new_score) {
		cout << "   ADJUSTING BY: " << new_score << endl;
		edge_visits += 1;
		edge_total_score += new_score;
	}
};

struct MCTSNode {
	Position board;
	bool evals_populated = false;
	Evaluations evals;
	shared_ptr<MCTSNode> parent;
	int all_edge_visits = 0;
	std::unordered_map<Move, MCTSEdge> outgoing_edges;

	MCTSNode(const Position& board) : board(board) {}

	double total_action_score(const Move& m) {
		assert(evals_populated);
		const auto it = outgoing_edges.find(m);
		double u_score, Q_score;
		if (it == outgoing_edges.end()) {
			u_score = sqrt(1 + all_edge_visits);
			Q_score = 0;
		} else {
			const MCTSEdge& edge = (*it).second;
			u_score = sqrt(1 + all_edge_visits) / (1 + edge.edge_visits);
			Q_score = edge.get_edge_score();
		}
		u_score *= exploration_parameter * evals.posterior.at(m);
//		cout << "Action score: " << this << " " << move_string(m) << " Q=" << Q_score << " u=" << u_score << " p=" << evals.posterior.at(m) << endl;
		return u_score + Q_score;
	}

	void populate_evals() {
		if (evals_populated)
			return;
		evals.populate(board);
		evals_populated = true;
	}

	Move select_action(bool use_dirichlet_noise) {
		populate_evals();
		// If we have no legal moves then return a null move.
		if (evals.posterior.size() == 0)
			return NO_MOVE;
		// If the game is over then return a null move.
		if (evals.game_over)
			return NO_MOVE;
		// Find the best move according to total_action_score.
		Move best_move = NO_MOVE;
		double best_score = -1;
		for (const auto p : evals.posterior) {
			double score = total_action_score(p.first);
			if (score > best_score) {
				best_move = p.first;
				best_score = score;
			}
		}
		return best_move;
	}
};

struct MCTS {
	shared_ptr<MCTSNode> root_node;
	bool use_dirichlet_noise;

	MCTS(const Position& root_board, bool use_dirichlet_noise)
		: use_dirichlet_noise(use_dirichlet_noise)
	{
		root_node = std::make_shared<MCTSNode>(root_board);
	}

	std::tuple<shared_ptr<MCTSNode>, Move, std::vector<MCTSEdge*>> select_principal_variation(bool best=false) {
		shared_ptr<MCTSNode> node = root_node;
		std::vector<MCTSEdge*> edges_on_path;
		Move move = NO_MOVE;
		while (true) {
			if (best) {
				// Pick the edge that has the highest visit count.
				auto it = std::max_element(
					node->outgoing_edges.begin(),
					node->outgoing_edges.end(),
					[](
						const std::pair<Move, MCTSEdge>& a,
						const std::pair<Move, MCTSEdge>& b
					) {
						return a.second.edge_visits < b.second.edge_visits;
					}
				);
				move = (*it).first;
			} else {
				// Pick the edge that has the current highest k-armed bandit value.
				move = node->select_action(use_dirichlet_noise);
			}
			// If the tree doesn't continue in the direction of this move, then break.
			const auto it = node->outgoing_edges.find(move);
			if (it == node->outgoing_edges.end())
				break;
			MCTSEdge& edge = (*it).second;
			edges_on_path.push_back(&edge);
			node = edge.child_node;
		}
//		cout << "Edges length: " << edges_on_path.size() << endl;
		return {node, move, edges_on_path};
	}

	void step() {
		// 1) Pick a path through the tree.
		auto triple = select_principal_variation();
		// Darn, I wish I had structured bindings already. :(
		shared_ptr<MCTSNode>         leaf_node     = std::get<0>(triple);
		Move                         move          = std::get<1>(triple);
		std::vector<MCTSEdge*>       edges_on_path = std::get<2>(triple);

		shared_ptr<MCTSNode> new_node;

		// 2) If the move is non-null then expand once at the leaf.
		if (move != NO_MOVE) {
			Position new_board = leaf_node->board;
			makemove(new_board, move);
			new_node = std::make_shared<MCTSNode>(new_board);
			auto pair_it_success = leaf_node->outgoing_edges.insert({
				move,
				MCTSEdge{move, leaf_node, new_node},
			});
			MCTSEdge& new_edge = (*pair_it_success.first).second;
			edges_on_path.push_back(&new_edge);
//			cout << "MAKING NOVEL MOVE: " << move_string(move) << endl;
		} else {
			// If the move is null, then we had no legal moves, and we just propagate the score again.
			// This occurs when we're repeatedly hitting a scored final position.
			new_node = leaf_node;
		}
		// Print out the explored path.
		for (auto& edge : edges_on_path) {
			cout << move_string(edge->edge_move) << " ";
		}
		cout << endl;

		// 3) Evaluate the new node.
		new_node->populate_evals();
		// Convert the expected value result into a score.
		double value_score = (new_node->evals.value + 1.0) / 2.0;
		// 4) Backup.
		bool inverted = false;
		for (auto it = edges_on_path.rbegin(); it != edges_on_path.rend(); it++) {
			MCTSEdge& edge = **it;
			inverted = not inverted;
			value_score = 1.0 - value_score;
			assert(inverted == (edge.parent_node->board.turn != new_node->board.turn));
			edge.adjust_score(value_score);
			edge.parent_node->all_edge_visits++;
		}
		assert(edges_on_path.size() != 0);
	}
};

int main() {
	global_rpc_connection = new rpc::client("127.0.0.1", 6500);

	Position board;
	set_board(board, "x5o/7/3-3/2-1-2/3-3/7/o5x x");

	MCTS mcts(board, false);
	for (int i = 0; i < 100; i++)
		mcts.step();
}

