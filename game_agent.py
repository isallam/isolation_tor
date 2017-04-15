"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass

# board position that provide the most possible next move, 8 in this case.
open_book_level1 = [(2,2), (2,3), (2,4), (3,2), (3,3), (3,4)]
num_elements_book_level1 = len(open_book_level1)

# board position that provide the 6 possible next move.
open_book_level2 = [(1,2), (1,3), (1,4), (2,1), (2, 5, (3, 1), (3,5), (4,1), (4,5), (5,2), (5,3), (5,4))]

# board positions that the least favourite since it only provide 2 possible next move
corner_spots = [(0,0), (0,6), (6,0), (6,6)]
num_elements_corner_spots = len(corner_spots)

# board positions that next least since they provide only 3 possible next move
only_three_options = [(0,1),(0,5), (1,0), (1,6), (5,0), (5,6), (6,1), (6,5)]

better_moves = open_book_level1 + open_book_level2
len_better_moves = len(better_moves)

worse_moves = corner_spots + only_three_options
len_worse_moves = len(worse_moves)

def pick_good_move(legal_moves):
    """ Check if we have a good move in the set of legal moves.
        pick one that is best suitable or return a random one if non
        exist in the open book

    Parameters
    ----------
    legal_moves: a list of legal moves

    returns
    tuple
        picked from the list

    """

    # check open_book_level1 which gives 8 possible next moves
    to_choose_from = list(set(open_book_level1) & set(legal_moves))
    if not to_choose_from:
        # check level2 which gives 6 possible next moves
        to_choose_from = list(set(open_book_level2) & set(legal_moves))

    if to_choose_from:
        return to_choose_from[random.randint(0, len(to_choose_from) -1)]
    
    #return legal_moves[random.randint(0, len(legal_moves) - 1)]
    return (-1, -1)

def multi_feature_score(game, player):
    """ Use various features of the game state to determine a score.
    features used:
        - number of player moves vs. opponent moves
        - intersection of the player's legal move set with the good position set
          and same for the opponenet.
        - number of empty spaces on the board

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : hashable
        One of the objects registered by the game object as a valid player.
        (i.e., `player` should be either game.__player_1__ or
        game.__player_2__).

    Returns
    ----------
    float
        The heuristic value of the current game state
    """
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    player_legal_moves = game.get_legal_moves(player)
    opp_player_legal_moves = game.get_legal_moves(game.get_opponent(player))

    own_moves = len(player_legal_moves)
    opp_moves = len(opp_player_legal_moves)
    if opp_moves == 0:
        return float('inf')
    if own_moves == 0:
        return float('-inf')

    num_player_better_moves = len(list(set(better_moves) & set(player_legal_moves)))
    num_player_worse_moves = len(list(set(worse_moves) & set(player_legal_moves)))
    num_opp_player_better_moves = len(list(set(better_moves) & set(opp_player_legal_moves)))
    num_opp_player_worse_moves = len(list(set(worse_moves) & set(opp_player_legal_moves)))
    num_available_spaces = len(game.get_blank_spaces())

    weight = float(game.move_count)/20.0
    value = float(own_moves - opp_moves) / float(own_moves + opp_moves)

    value += float(weight * num_player_better_moves - weight * num_opp_player_better_moves)/len_better_moves
    value += float(weight * num_opp_player_worse_moves - weight * num_player_worse_moves)/len_worse_moves

    # value += (50 - num_available_spaces) / 100.0
    # value += (game.move_count)/120.0

    return value


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """

    # try the improved_score() from the sample_players code base
    return multi_feature_score(game, player)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, data=None, timeout=1.):

        self.score = multi_feature_score
        self.time_left = None
        self.TIMER_THRESHOLD = max(timeout, 10.0)

    def get_move(self, game, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """

        self.time_left = time_left

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        legal_moves = game.get_legal_moves()

        if not legal_moves:
            return (-1, -1)

        # if we have only one legal move, we have no choice, just return it.
        if len(legal_moves) == 1:
            return legal_moves[0]

        # Check the open book again the legal_moves and keep one handy
        best_move = pick_good_move(legal_moves)
        if best_move != (-1, -1):
            return best_move
        else:
            best_move = legal_moves[random.randint(0, len(legal_moves) - 1)]

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            depth = 0;
            while True:
                depth += 1
                _, good_move = self.alphabeta(game, depth)
                if good_move != (-1, -1):
                    best_move = good_move
            pass

        except Timeout:
            # Handle any actions required at timeout, if necessary
            pass

        # Return the best move from the last completed search iteration
        return best_move


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):
        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        """
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if game.is_winner(self):
            return float('+inf'), game.get_legal_moves()

        legal_moves = game.get_legal_moves()

        # check if we reached the required depth or no more moves (terminal node)
        if (depth == 0) or (not legal_moves):
            game_score = self.score(game, self)
            return (game_score, (-1, -1))

        global gDepth
        good_choice = ()
        if maximizing_player:
            best_score = alpha
            good_choice = (alpha, (-1, -1))
            for move in legal_moves:
                game_branch = game.forecast_move(move)
                gDepth = depth
                branch_score, _ = self.alphabeta(game_branch, depth -1, alpha, beta, (not maximizing_player))
                if branch_score > best_score:
                    good_choice = (branch_score, move)
                    best_score = branch_score
                if best_score >= beta:
                    break
                alpha = max(alpha, best_score)

        else:
            best_score = beta
            good_choice = (beta, (-1, -1))
            for move in legal_moves:
                game_branch = game.forecast_move(move)
                branch_score, _ = self.alphabeta(game_branch, depth -1, alpha, beta, (not maximizing_player))
                if branch_score < best_score:
                    good_choice = (branch_score, move)
                    best_score = branch_score
                if best_score <= alpha:
                    good_choice = (best_score, move)
                    break
                beta = min(beta, best_score)

        return good_choice;
