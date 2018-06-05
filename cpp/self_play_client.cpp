// Game generation RPC client in C++.

#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <list>
#include <queue>
#include <chrono>
#include <random>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <cmath>
#include <cassert>

//#include <rpc/client.h>
#include <json.hpp>
#include "movegen.hpp"
#include "makemove.hpp"
#include "other.hpp"

#define STARTING_GAME_POSITION "x5o/7/3-3/2-1-2/3-3/7/o5x x"

using json = nlohmann::json;
using std::shared_ptr;
using std::cout;
using std::endl;

constexpr double exploration_parameter = 1.0;
constexpr double dirichlet_alpha = 0.15;
constexpr double dirichlet_weight = 0.25;
constexpr int maximum_game_plies = 400;
constexpr int steps_per_move = 400;
const std::vector<double> opening_randomization_schedule {
	0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125,
};

std::random_device rd;
std::default_random_engine generator(rd());
//rpc::client* global_rpc_connection;

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

std::vector<int> serialize_board_for_json(const Position& board) {
	std::vector<int> result;
	for (int y = 0; y < 7; y++) {
		for (int x = 0; x < 7; x++) {
			int square = x + 7 * (6 - y);
			uint64_t mask = 1ull << square;
			bool cross_present   = board.pieces[PIECE::CROSS]  & mask;
			bool nought_present  = board.pieces[PIECE::NOUGHT] & mask;
			if (cross_present) {
				result.push_back(1);
			} else if (nought_present) {
				result.push_back(2);
			} else {
				result.push_back(0);
			}
		}
	}
	assert(result.size() == 7 * 7);
	return result;
}

int get_board_result(const Position& board, Move* optional_moves_buffer=nullptr) {
	int p1_pieces = popcountll(board.pieces[PIECE::CROSS]);
	int p2_pieces = popcountll(board.pieces[PIECE::NOUGHT]);
	int blockers  = popcountll(board.blockers);
	int empty_cells = 7 * 7 - p1_pieces - p2_pieces - blockers;
	// Assert that at least one player has any pieces.
	assert(not (p1_pieces == 0 and p2_pieces == 0));
	assert(p1_pieces + p2_pieces + blockers <= 7 * 7);
	// If either player has no pieces then the other player wins.
	if (p1_pieces == 0)
		return 2;
	if (p2_pieces == 0)
		return 1;
	// If the current player has no moves then adjudicate as though the other player owns every cell.
	Move moves_buffer[256];
	if (optional_moves_buffer == nullptr)
		optional_moves_buffer = moves_buffer;
	int num_moves = movegen(board, optional_moves_buffer);
	if (num_moves == 0) {
		if (board.turn == SIDE::CROSS)
			p2_pieces += empty_cells;
		if (board.turn == SIDE::NOUGHT)
			p1_pieces += empty_cells;
	}
	assert(p1_pieces + p2_pieces + blockers <= 7 * 7);
	// If the board is full then adjudicate a win by who has more pieces.
	if (p1_pieces + p2_pieces + blockers == 7 * 7) {
		assert(p1_pieces != p2_pieces);
		return p1_pieces < p2_pieces ? 2 : 1;
	}
	// Finally, if none of the above cases matched, then the game isn't finished yet.
	return 0;
}

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string);

struct Evaluations {
	bool game_over;
	double value;
	std::unordered_map<Move, double> posterior;

	void populate(int thread_id, const Position& board, bool use_dirichlet_noise) {
		// Compeltely reset the evaluation.
		posterior.clear();
		game_over = false;

		// Build a features map, initialized to all zeros.
		float feature_buffer[7 * 7 * 4] = {};
		for (int y = 0; y < 7; y++) {
			for (int x = 0; x < 7; x++) {
				// Fill in layer 0 with all ones.
				feature_buffer[stride_index<7, 7, 4>(x, y, 0)] = 1.0;

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
						feature_buffer[stride_index<7, 7, 4>(x, y, 1)] = 1.0;
					else
						feature_buffer[stride_index<7, 7, 4>(x, y, 2)] = 1.0;
				}
				// If there's a blocker write to layer 3.
				if (blocker_present)
					feature_buffer[stride_index<7, 7, 4>(x, y, 3)] = 1.0;
			}
		}

		// Call the RPC CNN evaluator.
//		std::string feature_string(reinterpret_cast<char*>(feature_buffer), sizeof(feature_buffer));
		auto request_result = request_evaluation(thread_id, feature_buffer);
//		auto result = global_rpc_connection->call("network", feature_string).as<std::tuple<std::string, double>>();
//		std::string& posterior_str = std::get<0>(result);
//		value = std::get<1>(result);
//		std::string posterior_str = "asdf";
//		value = 0.1;

//		const float* posterior_array = reinterpret_cast<const float*>(posterior_str.c_str());

		const float* posterior_array = request_result.first;
		value = request_result.second;
//		for (int i = 0; i < 7 * 7 * 17; i++)
//			assert(posterior_array[i] == 0.0);
//		assert(value == 0.0);

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
		Move moves_buffer[256];
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

		// Add Dirichlet noise.
		if (use_dirichlet_noise) {
			// TODO: Fill this in!
		}
	}
};

struct MCTSEdge;
struct MCTSNode;

struct MCTSEdge {
	Move edge_move;
	MCTSNode* parent_node;
	shared_ptr<MCTSNode> child_node;
	double edge_visits = 0;
	double edge_total_score = 0;

	MCTSEdge(Move edge_move, MCTSNode* parent_node, shared_ptr<MCTSNode> child_node)
		: edge_move(edge_move), parent_node(parent_node), child_node(child_node) {}

	double get_edge_score() const {
		if (edge_visits == 0)
			return 0;
		return edge_total_score / edge_visits;
	}

	void adjust_score(double new_score) {
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
		return u_score + Q_score;
	}

	void populate_evals(int thread_id, bool use_dirichlet_noise=false) {
		if (evals_populated)
			return;
		evals.populate(thread_id, board, use_dirichlet_noise);
		evals_populated = true;
	}

	Move select_action() {
		assert(evals_populated);
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
	int thread_id;
	Position root_board;
	bool use_dirichlet_noise;
	shared_ptr<MCTSNode> root_node;

	MCTS(int thread_id, const Position& root_board, bool use_dirichlet_noise)
		: thread_id(thread_id), root_board(root_board), use_dirichlet_noise(use_dirichlet_noise)
	{
		init_from_scratch(root_board);
	}

	void init_from_scratch(const Position& root_board) {
		root_node = std::make_shared<MCTSNode>(root_board);
		root_node->populate_evals(thread_id, use_dirichlet_noise);
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
				move = node->select_action();
			}
			// If the tree doesn't continue in the direction of this move, then break.
			const auto it = node->outgoing_edges.find(move);
			if (it == node->outgoing_edges.end())
				break;
			MCTSEdge& edge = (*it).second;
			edges_on_path.push_back(&edge);
			node = edge.child_node;
		}
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
				MCTSEdge{move, leaf_node.get(), new_node},
			});
			MCTSEdge& new_edge = (*pair_it_success.first).second;
			edges_on_path.push_back(&new_edge);
		} else {
			// If the move is null, then we had no legal moves, and we just propagate the score again.
			// This occurs when we're repeatedly hitting a scored final position.
			new_node = leaf_node;
		}

//		// Print out the explored path.
//		for (auto& edge : edges_on_path) {
//			cout << move_string(edge->edge_move) << " ";
//		}
//		cout << endl;

		// 3) Evaluate the new node to get a score to propagate back up the tree.
		new_node->populate_evals(thread_id);
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
		if (edges_on_path.size() == 0) {
			cout << ">>> No edges on path!" << endl;
			print(root_board, false);
			cout << ">>> Move:" << move_string(move) << endl;
		}
		assert(edges_on_path.size() != 0);
	}

	void play(Move move) {
		// See if the move is in our root node's outgoing edges.
		auto it = root_node->outgoing_edges.find(move);
		// If we miss, just throw away everything and init from scratch.
		if (it == root_node->outgoing_edges.end()) {
//			cout << "Miss!" << endl;
			makemove(root_board, move);
			init_from_scratch(root_board);
			return;
		}
		// Otherwise, reuse a subtree.
		root_node = (*it).second.child_node;
		root_board = root_node->board;
//		cout << "Hit a subtree with: " << root_node->all_edge_visits << endl;
	}
};

Move sample_proportionally_to_visits(const shared_ptr<MCTSNode>& node) {
	double x = std::uniform_real_distribution<float>{0, 1}(generator);
	for (const std::pair<Move, MCTSEdge>& p : node->outgoing_edges) {
		double weight = p.second.edge_visits / node->all_edge_visits;
		if (x <= weight)
			return p.first;
		x -= weight;
	}
	// If we somehow slipped through then return some arbitrary element.
	std::cerr << "Potential bug: Weird numerical edge case in sampling!" << endl;
	return (*node->outgoing_edges.begin()).first;
}

json generate_game(int thread_id) {
	Position board;
	set_board(board, STARTING_GAME_POSITION);
	MCTS mcts(thread_id, board, true);

	json entry = {{"boards", {}}, {"moves", {}}};

	for (unsigned int ply = 0; ply < maximum_game_plies; ply++) {
		// Do a number of steps.
		for (int step = 0; step < steps_per_move; step++)
			mcts.step();
		// Sample a move according to visit counts.
		Move selected_move = sample_proportionally_to_visits(mcts.root_node);
		Move training_move = selected_move;
		// If appropriate choose a uniformly random legal move in the opening.
		if (ply < opening_randomization_schedule.size() and
		    std::uniform_real_distribution<double>{0, 1}(generator) < opening_randomization_schedule[ply]) {
			// Pick a random move.
			//cout << "Picking random move." << endl;
			int random_index = std::uniform_int_distribution<int>{0, static_cast<int>(mcts.root_node->evals.posterior.size()) - 1}(generator);
			auto it = mcts.root_node->evals.posterior.begin();
			std::advance(it, random_index);
			selected_move = (*it).first;
		}
//		cout << "Move: " << move_string(selected_move) << endl;
		entry["boards"].push_back(serialize_board_for_json(mcts.root_board));
		entry["moves"].push_back(move_string(training_move));
		mcts.play(selected_move);
		if (get_board_result(mcts.root_node->board) != 0)
			break;
	}
	entry["result"] = get_board_result(mcts.root_node->board);
	return entry;
}

// ================================================
//        T h r e a d e d   W o r k l o a d
// ================================================

struct Worker;
struct ResponseSlot;

std::list<Worker> global_workers;
std::vector<Worker*> global_workers_by_id;
float* global_fill_buffers[2];
int global_buffer_entries;

int current_buffer = 0;
int fill_levels[2] = {0, 0};
std::vector<ResponseSlot> response_slots[2];
std::queue<int> global_filled_queue;
std::mutex global_mutex;

struct ResponseSlot {
	int thread_id;
};

struct Worker {
	std::mutex thread_mutex;
	std::condition_variable cv;
	std::thread t;

	bool response_filled;
	double response_value;
	float response_posterior[7 * 7 * 17];

	Worker(int thread_id)
		: t(Worker::thread_main, thread_id) {}

	static void thread_main(int thread_id) {
//		cout << "Launching thread: " << thread_id << endl;
		while (true) {
			json game = generate_game(thread_id);
			if (game["result"] == 0) {
				cout << "Skipping game with null result." << endl;
				continue;
			}
			cout << thread_id << " Game generated. Plies: " << game["moves"].size() << endl;
		}
	}
};

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string) {
	// Write an entry into the appropriate work queue.
	{
		std::lock_guard<std::mutex> global_lock(global_mutex);
		int slot_index = fill_levels[current_buffer]++;
		// Copy our features into the big buffer.
		float* destination = global_fill_buffers[current_buffer] + (7 * 7 * 4) * slot_index;
		std::copy(feature_string, feature_string + (7 * 7 * 4), destination);
		// Place an entry requesting a reply.
		response_slots[current_buffer][slot_index].thread_id = thread_id;
		// Set that we're waiting on a response.
		Worker& worker = *global_workers_by_id[thread_id];
		worker.response_filled = false;
		// Swap buffers if we filled up the current one.
		if (fill_levels[current_buffer] == global_buffer_entries) {
//			cout << "Flip buffer from " << current_buffer << endl;
			global_filled_queue.push(current_buffer);
			current_buffer = 1 - current_buffer;
		}
		// TODO: Notify the main thread so it doesn't have to poll.
	}
	// Wait on a reply.
	Worker& worker = *global_workers_by_id[thread_id];
	std::unique_lock<std::mutex> lk(worker.thread_mutex);
	worker.cv.wait(lk, [&worker]{ return worker.response_filled; });
	// Response collected!
	return {worker.response_posterior, worker.response_value};
}

extern "C" void launch_threads(float* fill_buffer1, float* fill_buffer2, int buffer_entries, int thread_count) {
	global_fill_buffers[0] = fill_buffer1;
	global_fill_buffers[1] = fill_buffer2;
	global_buffer_entries = buffer_entries;
	cout << "Launching into " << fill_buffer1 << ", " << fill_buffer2 << " with " << buffer_entries << " entries and " << thread_count << " threads." << endl;

	for (int i = 0; i < buffer_entries; i++) {
		response_slots[0].push_back(ResponseSlot());
		response_slots[1].push_back(ResponseSlot());
	}

	for (int i = 0; i < thread_count; i++) {
		global_workers.emplace_back(i);
		global_workers_by_id.push_back(&global_workers.back());
	}
//	for (std::thread& t : global_workers)
//		t.join();
}

extern "C" int get_workload() {
	while (true) {
		{
			// Check if a workload is ready.
			std::lock_guard<std::mutex> global_lock(global_mutex);
//			cout << "Levels: " << fill_levels[0] << " " << fill_levels[1] << endl;
//			if (fill_levels[0] == global_buffer_entries)
//				return 0;
//			if (fill_levels[1] == global_buffer_entries)
//				return 1;
			if (not global_filled_queue.empty()) {
				int workload_index = global_filled_queue.front();
				global_filled_queue.pop();
				return workload_index;
			}
		}
		std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}
}

extern "C" void complete_workload(int workload, float* posteriors, float* values) {
	std::lock_guard<std::mutex> global_lock(global_mutex);
	for (int i = 0; i < global_buffer_entries; i++) {
		ResponseSlot& slot = response_slots[workload][i];
		Worker& worker = *global_workers_by_id[slot.thread_id];
		worker.response_value = values[i];
		std::copy(posteriors, posteriors + (7 * 7 * 17), worker.response_posterior);
		posteriors += (7 * 7 * 17);
		{
			std::lock_guard<std::mutex> lk(worker.thread_mutex);
			worker.response_filled = true;
		}
		worker.cv.notify_one();
	}
	fill_levels[workload] = 0;
}

#if 0
int main(int argc, const char** argv) {
	int my_number = std::stoi(argv[1]);
	int port = std::stoi(argv[2]);
	global_rpc_connection = new rpc::client("127.0.0.1", port);

	std::string path = "games/self-play-";
	for (int i = 0; i < 16; i++)
		path += "0123456789abcdef"[std::uniform_int_distribution<int>{0, 15}(generator)];
	path += ".json";

	cout << my_number << " on port " << port << " writing to: " << path << endl;

	std::ofstream out(path);

	while (true) {
		json game = generate_game();
		if (game["result"] == 0) {
			cout << "Skipping game with null result." << endl;
			continue;
		}
		out << game << "\n";
		out.flush();
		cout << my_number << " Game generated." << endl;
	}
}
#endif

