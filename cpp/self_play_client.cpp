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
#include <atomic>
#include <cmath>
#include <cassert>

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
//const std::vector<double> opening_randomization_schedule {
//	0.2, 0.2, 0.1, 0.1, 0.05, 0.05, 0.025, 0.025, 0.0125, 0.0125,
//};

std::random_device rd;
std::default_random_engine generator(rd());

// We raise this exception in worker threads when they're done.
struct StopWorking : public std::exception {};

// Configuration.
int global_visits;

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

int get_board_result(const Position& board, Move* optional_moves_buffer=nullptr, int* optional_moves_count=nullptr) {
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
	if (optional_moves_count != nullptr)
		*optional_moves_count = num_moves;
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
	assert(num_moves != 0); // If there are no moves then we must adjudicate one way or the other.
	// Finally, if none of the above cases matched, then the game isn't finished yet.
	return 0;
}

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string);

struct Evaluations {
	bool game_over;
	double value;
	std::unordered_map<Move, double> posterior;

	void populate(int thread_id, const Position& board, bool use_dirichlet_noise) {
		// Completely reset the evaluation.
		posterior.clear();
		game_over = false;

		Move moves_buffer[256];
		int num_moves;
		int result = get_board_result(board, moves_buffer, &num_moves);

		if (result != 0) {
			game_over = true;
			// Score from the perspective of the current player.
			assert(result == 1 or result == 2);
			value = result == 1 ? 1.0 : -1.0;
			// Flip the value to be from the perspective of the current player.
			// TODO: XXX: Validate that I got this the right way around.
			if (board.turn == SIDE::NOUGHT)
				value *= -1;
			return;
		}

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

		auto request_result = request_evaluation(thread_id, feature_buffer);
		const float* posterior_array = request_result.first;
		value = request_result.second;

		// Softmax the posterior array.
		double softmaxed[7 * 7 * 17];
		for (int i = 0; i < (7 * 7 * 17); i++)
			softmaxed[i] = exp(posterior_array[i]);
		double total = 0.0;
		for (int i = 0; i < (7 * 7 * 17); i++)
			total += softmaxed[i];
		if (total != 0.0) {
			for (int i = 0; i < (7 * 7 * 17); i++)
				softmaxed[i] /= total;
		}

		// Evaluate movegen.
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
		// Normalize the posterior.
		if (total_probability != 0.0) {
			for (auto& p : posterior)
				p.second /= total_probability;
		}

		assert(num_moves > 0);
		assert(posterior.size() > 0);

		// Add Dirichlet noise.
		if (use_dirichlet_noise) {
			std::gamma_distribution<double> distribution(dirichlet_alpha, 1.0);
			std::vector<double> dirichlet_distribution;
			for (auto& p : posterior)
				dirichlet_distribution.push_back(distribution(generator));
			// Normalize the Dirichlet distribution.
			double total = 0.0;
			for (double x : dirichlet_distribution)
				total += x;
			for (double& x : dirichlet_distribution)
				x /= total;
			// Perform a weighted sum of this Dirichlet distribution and our previous posterior.
			int index = 0;
			for (auto& p : posterior)
				p.second = dirichlet_weight * dirichlet_distribution[index++] + (1.0 - dirichlet_weight) * p.second;
			// Assert approximate normalization.
			double test_total = 0.0;
			for (auto& p : posterior)
				test_total += p.second;
//			assert(0.99 < test_total and test_total < 1.01);
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
		bool flag = false;
		for (const auto p : evals.posterior) {
			double score = total_action_score(p.first);
			if (p.first == NO_MOVE) {
				std::cerr << "Bad posterior had NO_MOVE in moves list." << endl;
				std::cerr << "Posterior length: " << evals.posterior.size() << endl;
				std::cerr << "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!" << endl;
			}
			assert(not std::isnan(score));
			assert(score >= 0.0);
			if (score >= best_score) {
				best_move = p.first;
				best_score = score;
				flag = true;
			}
		}
		if (best_move == NO_MOVE) {
			std::cerr << "                  WARNING Best move is no move!" << endl;
			std::cerr << "       Posterior size: " << evals.posterior.size() << endl;
			std::cerr << "     Flag status: " << flag << " END FLAG" << endl;
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
			print(root_board, true);
			cout << "Valid moves list:" << endl;
			print_moves(root_board);
			cout << "Select action at root:" << endl;
			Move selected_action = root_node->select_action();
			cout << "Posterior length:" << root_node->evals.posterior.size() << endl;
			cout << "Game over:" << root_node->evals.game_over << endl;
			cout << "Here it is: " << selected_action.from << " " << selected_action.to << endl;
			cout << ">>> Move:" << move.from << " " << move.to << endl;
		}
		assert(edges_on_path.size() != 0);
	}

	void play(Move move) {
		// See if the move is in our root node's outgoing edges.
		auto it = root_node->outgoing_edges.find(move);
		// If we miss, just throw away everything and init from scratch.
		if (it == root_node->outgoing_edges.end()) {
			makemove(root_board, move);
			init_from_scratch(root_board);
			return;
		}
		// Otherwise, reuse a subtree.
		root_node = (*it).second.child_node;
		root_board = root_node->board;
		// Now that a new node is the root we have to redo its evals with Dirichlet noise, if required.
		// This is a little wasteful, when we could just apply Dirichlet noise, but it's not that bad.
		root_node->evals_populated = false;
		root_node->populate_evals(thread_id, use_dirichlet_noise);

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
	json entry = {{"boards", {}}, {"moves", {}}, {"dists", {}}};
	int steps_done = 0;

	for (unsigned int ply = 0; ply < maximum_game_plies; ply++) {
		// Do a number of steps.
		while (mcts.root_node->all_edge_visits < global_visits) {
			mcts.step();
			steps_done++;
		}
		// Sample a move according to visit counts.
		Move selected_move = sample_proportionally_to_visits(mcts.root_node);
		Move training_move = selected_move;

/*
		// If appropriate choose a uniformly random legal move in the opening.
		if (ply < opening_randomization_schedule.size() and
		    std::uniform_real_distribution<double>{0, 1}(generator) < opening_randomization_schedule[ply]) {
			// Pick a random move.
			int random_index = std::uniform_int_distribution<int>{0, static_cast<int>(mcts.root_node->evals.posterior.size()) - 1}(generator);
			auto it = mcts.root_node->evals.posterior.begin();
			std::advance(it, random_index);
			selected_move = (*it).first;
		}
*/
		entry["boards"].push_back(serialize_board_for_json(mcts.root_board));
		entry["moves"].push_back(move_string(training_move));
		entry["dists"].push_back({});
		// Write out the entire visit distribution.
		for (const std::pair<Move, MCTSEdge>& p : mcts.root_node->outgoing_edges) {
			double weight = p.second.edge_visits / mcts.root_node->all_edge_visits;
			entry["dists"].back()[move_string(p.first)] = weight;
		}
		mcts.play(selected_move);
		if (get_board_result(mcts.root_node->board) != 0)
			break;
	}
//	float work_factor = steps_done / (float)(global_visits * entry["moves"].size());
//	cout << "Work factor: " << work_factor << endl;

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
std::ofstream* global_output_file;

int current_buffer = 0;
int fill_levels[2] = {0, 0};
std::vector<ResponseSlot> response_slots[2];
std::queue<int> global_filled_queue;
std::mutex global_mutex;
std::atomic<bool> keep_working;

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
		while (true) {
			json game;
			try {
				game = generate_game(thread_id);
			} catch (StopWorking& e) {
				return;
			}
			if (game["result"] == 0) {
				cout << "Skipping game with null result." << endl;
				continue;
			}
			{
				std::lock_guard<std::mutex> global_lock(global_mutex);
				cout << thread_id << " Game generated. Plies: " << game["moves"].size() << endl;
				(*global_output_file) << game << "\n";
				global_output_file->flush();
			}
		}
	}
};

std::pair<const float*, double> request_evaluation(int thread_id, const float* feature_string) {
	// Write an entry into the appropriate work queue.
	{
		std::lock_guard<std::mutex> global_lock(global_mutex);
		int slot_index = fill_levels[current_buffer]++;
		assert(0 <= slot_index and slot_index < response_slots[0].size());
		// Copy our features into the big buffer.
		float* destination = global_fill_buffers[current_buffer] + (7 * 7 * 4) * slot_index;
		std::copy(feature_string, feature_string + (7 * 7 * 4), destination);
		// Place an entry requesting a reply.
		response_slots[current_buffer].at(slot_index).thread_id = thread_id;
		// Set that we're waiting on a response.
		Worker& worker = *global_workers_by_id[thread_id];
		worker.response_filled = false;
		// Swap buffers if we filled up the current one.
		if (fill_levels[current_buffer] == global_buffer_entries) {
			global_filled_queue.push(current_buffer);
			current_buffer = 1 - current_buffer;
		}
		// TODO: Notify the main thread so it doesn't have to poll.
	}
	// Wait on a reply.
	Worker& worker = *global_workers_by_id[thread_id];
	std::unique_lock<std::mutex> lk(worker.thread_mutex);
	while (not worker.response_filled) {
		worker.cv.wait_for(lk, std::chrono::milliseconds(250), [&worker]{
			return worker.response_filled;
		});
		if (not keep_working)
			throw StopWorking();
	}
	// Response collected!
	return {worker.response_posterior, worker.response_value};
}

extern "C" void launch_threads(char* output_path, int visits, float* fill_buffer1, float* fill_buffer2, int buffer_entries, int thread_count) {
	global_visits = visits;
	global_fill_buffers[0] = fill_buffer1;
	global_fill_buffers[1] = fill_buffer2;
	global_buffer_entries = buffer_entries;
	cout << "Launching into " << fill_buffer1 << ", " << fill_buffer2 << " with " << buffer_entries << " entries and " << thread_count << " threads." << endl;

	cout << "Writing to: " << output_path << endl;
	global_output_file = new std::ofstream(output_path, std::ios_base::app);
	keep_working = true;

	for (int i = 0; i < buffer_entries; i++) {
		response_slots[0].push_back(ResponseSlot());
		response_slots[1].push_back(ResponseSlot());
	}

	{
		std::lock_guard<std::mutex> global_lock(global_mutex);
		for (int i = 0; i < thread_count; i++) {
			global_workers.emplace_back(i);
			global_workers_by_id.push_back(&global_workers.back());
		}
	}
}

extern "C" int get_workload(void) {
	while (true) {
		{
			std::lock_guard<std::mutex> global_lock(global_mutex);
			// Check if a workload is ready.
			if (not global_filled_queue.empty()) {
				int workload_index = global_filled_queue.front();
				global_filled_queue.pop();
				return workload_index;
			}
		}
		std::this_thread::sleep_for(std::chrono::microseconds(100));
	}
}

extern "C" void complete_workload(int workload, float* posteriors, float* values) {
	std::lock_guard<std::mutex> global_lock(global_mutex);
	for (int i = 0; i < global_buffer_entries; i++) {
		ResponseSlot& slot = response_slots[workload].at(i);
		Worker& worker = *global_workers_by_id.at(slot.thread_id);
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

extern "C" void shutdown(void) {
	keep_working = false;
	for (Worker& w : global_workers)
		w.t.join();
	// Clear out data structures to setup for another run.
	for (int i : {0, 1})
		response_slots[i].clear();
	global_workers.clear();
	global_workers_by_id.clear();
}

