import random
import numpy as np
from collections import defaultdict
from rich import print
from rich.panel import Panel

class TicTacToeGame:
    """
    Defines the Tic-Tac-Toe game environment.
    """
    def __init__(self):
        # Initialize empty 3x3 board (represented as a list of 9 elements)
        self.reset()

    def reset(self):
        """Reset the game board to empty state."""
        self.board = [' ' for _ in range(9)]
        self.current_player = 'X'  # X always starts
        self.winner = None
        self.game_over = False
        return self.get_state()

    def get_state(self):
        """Return the current state of the board as a string."""
        return ''.join(self.board)

    def display_board(self):
        """Display the current game board."""
        print("\n[bold blue]Current Board:[/bold blue]")
        for i in range(0, 9, 3):
            print(f" {self.board[i]} | {self.board[i+1]} | {self.board[i+2]} ")
            if i < 6:
                print("-----------")
        print()

    def available_moves(self):
        """Return a list of indexes of available moves (empty cells)."""
        return [i for i, spot in enumerate(self.board) if spot == ' ']

    def make_move(self, position, player=None):
        """
        Make a move on the board.
        Returns True if the move is valid, False otherwise.
        """
        if player is None:
            player = self.current_player

        if position not in self.available_moves():
            return False

        self.board[position] = player

        # Check for win or tie
        if self.check_win(player):
            self.winner = player
            self.game_over = True
        elif self.check_tie():
            self.game_over = True
        else:
            # Switch player
            self.current_player = 'O' if self.current_player == 'X' else 'X'

        return True

    def check_win(self, player):
        """Check if the specified player has won."""
        # Check rows
        for i in range(0, 9, 3):
            if self.board[i] == self.board[i+1] == self.board[i+2] == player:
                return True

        # Check columns
        for i in range(3):
            if self.board[i] == self.board[i+3] == self.board[i+6] == player:
                return True

        # Check diagonals
        if self.board[0] == self.board[4] == self.board[8] == player:
            return True
        if self.board[2] == self.board[4] == self.board[6] == player:
            return True

        return False

    def check_tie(self):
        """Check if the game is a tie."""
        return ' ' not in self.board and self.winner is None


class SimpleAIAgent:
    """
    A simple AI agent that can learn to play Tic-Tac-Toe using a basic
    implementation of the Monte Carlo learning method.
    """
    def __init__(self, epsilon=0.1):
        self.epsilon = epsilon  # Exploration rate
        self.state_values = defaultdict(float)  # Store states and their values
        self.state_visits = defaultdict(int)    # Count state visits

    def choose_action(self, game, player):
        """
        Choose an action using an epsilon-greedy strategy.
        """
        available = game.available_moves()

        # Exploration: random move
        if random.random() < self.epsilon:
            return random.choice(available)

        # Exploitation: choose best move based on learned values
        best_value = -float('inf')
        best_moves = []

        for move in available:
            # Create a copy of the board
            board_copy = game.board.copy()

            # Simulate the move
            board_copy[move] = player
            next_state = ''.join(board_copy)

            # Get the value of this state
            value = self.state_values[next_state]

            # Update best value and moves
            if value > best_value:
                best_value = value
                best_moves = [move]
            elif value == best_value:
                best_moves.append(move)

        # If no good move found or all have same value, pick randomly
        if not best_moves:
            return random.choice(available)

        return random.choice(best_moves)

    def learn(self, state_history, reward):
        """
        Update state values based on game outcome.
        A simple implementation of Monte Carlo learning.
        """
        for state in state_history:
            self.state_visits[state] += 1
            # Update value estimate using incremental mean
            self.state_values[state] += (reward - self.state_values[state]) / self.state_visits[state]


def train_agent(agent, episodes=10000):
    """Train the agent by playing against itself."""
    game = TicTacToeGame()

    for episode in range(episodes):
        # Reset the game
        game.reset()

        # Keep track of states for learning
        x_states = []
        o_states = []

        while not game.game_over:
            # Get current state
            current_state = game.get_state()

            # Store state for learning
            if game.current_player == 'X':
                x_states.append(current_state)
            else:
                o_states.append(current_state)

            # Choose and make move
            move = agent.choose_action(game, game.current_player)
            game.make_move(move)

        # Learning from game outcome
        if game.winner == 'X':
            agent.learn(x_states, 1.0)  # Reward for winning
            agent.learn(o_states, 0.0)  # Penalty for losing
        elif game.winner == 'O':
            agent.learn(x_states, 0.0)  # Penalty for losing
            agent.learn(o_states, 1.0)  # Reward for winning
        else:
            agent.learn(x_states, 0.5)  # Draw
            agent.learn(o_states, 0.5)  # Draw

        # Print progress
        if (episode + 1) % 1000 == 0:
            return None
            #print(f"Completed {episode + 1} training episodes")


def get_human_move(game):
    """Get move from human player."""
    while True:
        try:
            move = int(input(f"Enter your move (1-9): ")) - 1
            if 0 <= move <= 8 and move in game.available_moves():
                return move
            else:
                print("Invalid move. Please enter a number between 1-9 for an empty space.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def play_game(agent):
    """Main game loop for human vs. AI."""
    game = TicTacToeGame()

    # Decide who goes first
    human_player = input("Do you want to be X (goes first) or O? ").upper()
    if human_player not in ['X', 'O']:
        human_player = 'X'  # Default to X

    ai_player = 'O' if human_player == 'X' else 'X'

    print(f"\nYou are {human_player}. AI is {ai_player}.")
    print("Board positions are numbered 1-9 from top-left to bottom-right.")

    game.display_board()

    while not game.game_over:
        if game.current_player == human_player:
            # Human's turn
            move = get_human_move(game)
            game.make_move(move)
        else:
            # AI's turn
            print("AI is thinking...")
            move = agent.choose_action(game, ai_player)
            game.make_move(move)
            print(f"AI placed {ai_player} at position {move + 1}")

        game.display_board()

    # Game over
    if game.winner:
        if game.winner == human_player:
            print("[bold green]Congratulations! You won![/bold green]")
        else:
            print("[bold red]AI wins![/bold red]")
    else:
        print("[bold yellow]It's a tie![/bold yellow]")


def main():
    # Create the AI agent
    ai_agent = SimpleAIAgent(epsilon=0.1)

    # Game instructions
    game_instructions = '''\n
[bold black on white]Welcome to Tic-Tac-Toe with Learning AI![/bold black on white]

[bold black on white]This classic game is played on a 3x3 grid where players take turns placing their mark (X or O).[/bold black on white]

[bold black on white]Game Rules:[/bold black on white]
[bold black on white]1. The game is played on a 3x3 grid.[/bold black on white]
[bold black on white]2. You can choose to be X (goes first) or O.[/bold black on white]
[bold black on white]3. Players take turns placing their marks in empty squares.[/bold black on white]
[bold black on white]4. The first player to get three marks in a row (horizontally, vertically, or diagonally) wins.[/bold black on white]
[bold black on white]5. If all squares are filled and no player has three in a row, the game is a tie.[/bold black on white]

[bold black on white]Special Feature:[/bold black on white]
[bold black on white]The AI opponent can learn from its mistakes and improve its strategy over time![/bold black on white]

[bold black on white]Board Layout:[/bold black on white]
[bold black on white]1 | 2 | 3[/bold black on white]
[bold black on white]---------[/bold black on white]
[bold black on white]4 | 5 | 6[/bold black on white]
[bold black on white]---------[/bold black on white]
[bold black on white]7 | 8 | 9[/bold black on white]

[bold black on white]Enter the number corresponding to the position where you want to place your mark.[/bold black on white]
'''
    panel = Panel(game_instructions, title="[bold magenta]TIC-TAC-TOE GAME INSTRUCTIONS[/bold magenta]")
    print(panel)

    # Train the agent
    print("[bold green]Hold on, AI agent is getting ready... (this may take a moment)[/bold green]")
    train_agent(ai_agent, episodes=10000)
    print("[bold green]AI agent is ready![/bold green]")

    # Play game against trained agent
    while True:
        play_game(ai_agent)

        # Ask if the player wants to play again with limited attempts
        max_attempts = 4
        attempts = 0
        while attempts < max_attempts:
            play_again = input("\nDo you want to play again? (yes/no): ").lower()
            if play_again in ('yes', 'no'):
                break
            attempts += 1
            print(f"[bold red]Invalid answer. You have {max_attempts - attempts} more attempts.[/bold red]")

        if play_again == 'yes':
            print("\n[bold green]Great! Let's play again![/bold green]\n")
            continue
        elif play_again == 'no':
            print("[bold cyan]Thanks for playing! Goodbye![/bold cyan]")
            break
        else:
            # User exhausted attempts without a valid answer
            print("\n[bold red]Too many invalid responses. Ending the game automatically. Goodbye![/bold red]")
            break


if __name__ == "__main__":
    main()
