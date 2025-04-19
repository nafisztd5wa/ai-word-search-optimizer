import pygame
import random
import heapq
from collections import defaultdict
import math
import matplotlib.pyplot as plt
import numpy as np

# set this variable to True when we want to have the ai solver make the guesses
ai_enabled = True  # Set to True to test AI

# SETUP DISPLAY
pygame.init()
WIDTH, HEIGHT = 900, 650
WIN = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wordle")

# SYSTEM VARIABLES
FPS = 60
clock = pygame.time.Clock()

# COLORS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
LIGHT_GRAY = (211, 214, 218)
DARK_GRAY = (120, 124, 126)
GREEN = (106, 170, 100)
YELLOW = (201, 180, 88)

# FONTS
TITLE_FONT = pygame.font.SysFont("bahnschrift", 30)
LETTER_FONT = pygame.font.SysFont("bahnschrift", 16)
BOX_FONT = pygame.font.SysFont("bahnschrift", 30)
MSG_BOX_FONT = pygame.font.SysFont("bahnschrift", 20)


# GAME VARIABLES
def set_game_variables():
    global cursor, guesses
    cursor = 0
    guesses = 0


# LOAD TEXT
def load_text():
    global wordleWords, allWords
    with open("wordleWords.txt", "r") as f:
        wordleWords = f.read().split(",")
    with open("allWords.txt", "r") as f:
        allWords = f.read().split(",")


# PICK A RANDOM WORD
def pick_word():
    global word
    word = random.choice(wordleWords).upper()


# WORDLE BOXES CALCULATION
BOX_WIDTH = 55
BOX_GAP = 5
box_startx = round((WIDTH - (BOX_WIDTH + BOX_GAP) * 5) / 2)
box_starty = 110


def setup_boxes():
    global boxes
    boxes = []
    for i in range(30):
        x = box_startx + BOX_GAP * 6 + ((BOX_WIDTH + BOX_GAP) * (i % 5))
        y = box_starty + ((i // 5) * (BOX_GAP + BOX_WIDTH))
        boxes.append([x, y, '', WHITE])


# KEYBOARD KEYS CALCULATION
KEY_WIDTH = 45
KEY_GAP = 10
LKEY_WIDTH = KEY_WIDTH * 2 + KEY_GAP
key_startx = round((WIDTH - (KEY_WIDTH + KEY_GAP) * 9.5) / 2)
key_starty = 480
KEY_CHARS = ["Q", "W", "E", "R", "T", "Y", "U", "I", "O", "P", "A", "S", "D", "F", "G", "H", "J", "K", "L",
             "Z", "X", "C", "V", "B", "N", "M", "<<", "Enter"]


def setup_keys():
    global letters
    letters = []
    for i in range(28):
        if i == 27:
            x = letters[26][0] + KEY_WIDTH / 2 + KEY_GAP + LKEY_WIDTH / 2
        else:
            x = key_startx + KEY_GAP + ((KEY_WIDTH + KEY_GAP) * (i % 9.5))
        y = key_starty + ((i // 9.5) * (KEY_GAP + KEY_WIDTH))
        letters.append([x, y, KEY_CHARS[i], LIGHT_GRAY])


# SHOW POPUP MESSAGE
MSG_BOX_Y = 80
MSG_BOX_MARGIN = 12

# A* Search Solver
class AStarWordleSolver:
    def __init__(self, word_list):
        self.words = [w.upper() for w in word_list if len(w) == 5]
        self.constraints = []
        self.previous_guesses = set()

    def update_constraints(self, guess, feedback):
        self.constraints.append((guess, feedback))
        self.previous_guesses.add(guess)
        self._prune_domain()

    def _prune_domain(self):
        old_count = len(self.words)
        for guess, feedback in self.constraints:
            filtered_words = []
            for candidate in self.words:
                if self._matches_feedback(candidate, guess, feedback):
                    filtered_words.append(candidate)
            self.words = filtered_words
        print(f"A* pruned words from {old_count} to {len(self.words)}")
        if len(self.words) < 10:
            print("A* remaining words:", self.words)

    def _matches_feedback(self, candidate, guess, feedback):
        if len(candidate) != 5 or len(guess) != 5:
            return False

        cand_letters = list(candidate)
        guess_letters = list(guess)

        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if f == "green" and candidate[i] != g:
                return False
            if f == "green":
                cand_letters[i] = '*'
                guess_letters[i] = '*'

        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if g == '*':
                continue
            if f == "yellow":
                if g not in cand_letters:
                    return False
                cand_letters[cand_letters.index(g)] = '*'
            elif f == "gray":
                if g in cand_letters:
                    return False
        return True

    def _calculate_heuristic(self, word):
        if len(self.words) <= 2:
            return 0 if word in self.words else float('inf')

        letter_freq = defaultdict(int)
        pos_freq = defaultdict(lambda: defaultdict(int))

        for possible_word in self.words:
            for i, letter in enumerate(possible_word):
                letter_freq[letter] += 1
                pos_freq[i][letter] += 1

        score = 0
        seen_letters = set()

        for i, letter in enumerate(word):
            if letter not in seen_letters:
                freq = letter_freq[letter] / len(self.words)
                score += freq * (1 - freq)

                pos_score = pos_freq[i][letter] / len(self.words)
                score += pos_score * (1 - pos_score)

                seen_letters.add(letter)
            else:
                score *= 0.7

        if word not in self.words:
            score *= 0.9

        if word in self.previous_guesses:
            return float('inf')

        return -score

    def get_best_guess(self):
        if not self.words:
            return None
        if len(self.words) == 1:
            return self.words[0]

        open_set = []
        heapq.heappush(open_set, (0, "STARE"))  # Start with STARE if first guess
        visited = set()

        while open_set:
            current_cost, current_word = heapq.heappop(open_set)

            if current_word in visited:
                continue

            visited.add(current_word)

            if current_word in self.words:
                return current_word

            for word in self.words[:min(len(self.words), 100)]:  # Limit search space
                if word not in visited:
                    cost = self._calculate_heuristic(word)
                    heapq.heappush(open_set, (cost, word))

        return self.words[0] if self.words else None


def display_message(status):
    msg = ""
    if status == "won":
        if guesses == 0:
            msg = "Genius"
        elif guesses == 1:
            msg = "Magnificent"
        elif guesses == 2:
            msg = "Impressive"
        elif guesses == 3:
            msg = "Splendid"
        elif guesses == 4:
            msg = "Great"
        elif guesses == 5:
            msg = "Phew"
    elif status == "lost":
        msg = word
    elif status == "not_in_list":
        msg = "Not in word list"

    msg_box_text = MSG_BOX_FONT.render(msg, 1, WHITE)
    msg_box = pygame.Rect(WIDTH / 2 - msg_box_text.get_width() / 2 - MSG_BOX_MARGIN,
                          MSG_BOX_Y - msg_box_text.get_height() / 2 - MSG_BOX_MARGIN / 2,
                          msg_box_text.get_width() + MSG_BOX_MARGIN * 2,
                          msg_box_text.get_height() + MSG_BOX_MARGIN / 2 * 2)
    pygame.draw.rect(WIN, BLACK, msg_box, border_radius=3)
    WIN.blit(msg_box_text, (WIDTH / 2 - msg_box_text.get_width() / 2, MSG_BOX_Y - msg_box_text.get_height() / 2))

    pygame.display.update()

    if status == "not_in_list":
        pygame.time.delay(1000)
    else:
        pygame.time.delay(3000)


# CHECK GUESS
def check_guess():
    global ai_feedback
    feedback = ["gray"] * 5

    correct = 0
    word_letters = [*word]
    not_green = []
    for i in range(5):
        if boxes[5 * guesses + i][2] == word_letters[i]:
            boxes[5 * guesses + i][3] = GREEN
            feedback[i] = "green"
            word_letters[i] = ""
            correct += 1

            for letter in letters:
                x, y, ltr, color = letter
                if ltr == boxes[5 * guesses + i][2]:
                    letter[3] = GREEN
        else:
            not_green.append(i)

    for j in not_green:
        if boxes[5 * guesses + j][2] in word_letters:
            boxes[5 * guesses + j][3] = YELLOW
            feedback[j] = "yellow"
            word_letters.remove(boxes[5 * guesses + j][2])
            for letter in letters:
                x, y, ltr, color = letter
                if ltr == boxes[5 * guesses + j][2] and color == LIGHT_GRAY:
                    letter[3] = YELLOW
        else:
            boxes[5 * guesses + j][3] = DARK_GRAY
            for letter in letters:
                x, y, ltr, color = letter
                if ltr == boxes[5 * guesses + j][2] and color == LIGHT_GRAY:
                    letter[3] = DARK_GRAY

    print("Feedback: ", feedback)
    ai_feedback = feedback
    if correct == 5:
        return True
    return False


# DRAW
def draw():
    WIN.fill(WHITE)

    title_text = TITLE_FONT.render("Wordle", 1, BLACK)
    WIN.blit(title_text, (WIDTH / 2 - title_text.get_width() / 2, box_starty - 90))

    for i, letter in enumerate(letters):
        x, y, ltr, color = letter
        key_width = LKEY_WIDTH if i == 27 else KEY_WIDTH
        key = pygame.Rect(x - key_width / 2, y - KEY_WIDTH / 2, key_width, KEY_WIDTH)
        pygame.draw.rect(WIN, color, key, border_radius=4)
        letter_text = LETTER_FONT.render(ltr, 1, (BLACK if color == LIGHT_GRAY else WHITE))
        WIN.blit(letter_text, (x - letter_text.get_width() / 2, y - letter_text.get_height() / 2))

    for box in boxes:
        x, y, chr, color = box
        box_rect = pygame.Rect(x - BOX_WIDTH / 2, y - BOX_WIDTH / 2, BOX_WIDTH, BOX_WIDTH)
        if box[3] == WHITE:
            pygame.draw.rect(WIN, LIGHT_GRAY, box_rect, 2)
            box_text = BOX_FONT.render(chr, 1, BLACK)
        else:
            pygame.draw.rect(WIN, box[3], box_rect)
            box_text = BOX_FONT.render(chr, 1, WHITE)

        WIN.blit(box_text, (x - box_text.get_width() / 2, y - box_text.get_height() / 2))

    pygame.display.update()


bayesian_solver = None
minimax_solver = None


def setup_game():
    global run, ai_feedback, remaining_words, bayesian_solver, minimax_solver
    run = True
    set_game_variables()
    load_text()
    pick_word()
    print("Answer is: ", word)
    setup_keys()
    setup_boxes()
    ai_feedback = []
    remaining_words = wordleWords.copy()
    bayesian_solver = None
    minimax_solver = None


def ai_guess():
    # Hybrid Bayesian-Minimax approach
    global ai_feedback, remaining_words, guesses, bayesian_solver, minimax_solver
    print(f"\nMaking guess {guesses + 1}")
    if guesses == 0:
        bayesian_solver = BayesianWordleSolver(wordleWords)
        minimax_solver = MinimaxWordleSolver(wordleWords)
        return "STARE"
    current_guess = ""
    for i in range(5):
        current_guess += boxes[5 * (guesses - 1) + i][2]
    bayesian_solver.update_beliefs(current_guess, ai_feedback)
    minimax_solver.update_knowledge(current_guess, ai_feedback)
    if len(bayesian_solver.words) <= 1:
        return bayesian_solver.words[0]
    elif len(bayesian_solver.words) > 10:
        next_guess = bayesian_solver.get_best_guess()
        print(f"Using Bayesian solver, {len(bayesian_solver.words)} words remain")
    else:
        next_guess = minimax_solver.get_best_guess(6 - guesses)
        print(f"Using Minimax solver, {len(minimax_solver.words)} words remain")
    if next_guess is None:
        print("Warning: No valid guess found, using fallback")
        next_guess = "CRANE"
    print(f"Choosing word: {next_guess}")
    return next_guess
class BayesianWordleSolver:
    def __init__(self, word_list):
        self.words = [w.upper() for w in word_list if len(w) == 5]
        self.previous_guesses = []
        self._update_probabilities()
    def _update_probabilities(self):
        self.letter_freq = defaultdict(int)
        self.pos_freq = defaultdict(lambda: defaultdict(int))
        for word in self.words:
            for i, letter in enumerate(word):
                self.letter_freq[letter] += 1
                self.pos_freq[i][letter] += 1
    def update_beliefs(self, guess, feedback):
        print(f"Updating beliefs for guess: {guess} with feedback: {feedback}")
        old_count = len(self.words)
        self.previous_guesses.append((guess, feedback))
        filtered_words = self.words.copy()
        for prev_guess, prev_feedback in self.previous_guesses:
            filtered_words = [word for word in filtered_words
                              if self._matches_feedback(word, prev_guess, prev_feedback)]
        self.words = filtered_words
        print(f"Filtered words from {old_count} to {len(self.words)}")
        if len(self.words) < 10:
            print(f"Remaining words: {self.words}")
        self._update_probabilities()
    def _matches_feedback(self, word, guess, feedback):
        if len(word) != 5 or len(guess) != 5:
            return False
        word_letters = list(word)
        guess_letters = list(guess)
        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if f == "green" and word[i] != g:
                return False
            elif f == "green":
                word_letters[i] = '*'
                guess_letters[i] = '*'
        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if g == '*':
                continue
            if f == "yellow":
                if g not in word_letters:
                    return False
                word_letters[word_letters.index(g)] = '*'
            elif f == "gray" and g in word_letters:
                return False
        return True
    def get_best_guess(self):
        if not self.words:
            return None
        if len(self.words) <= 2:
            return self.words[0]
        best_score = float('-inf')
        best_word = None
        candidates = self.words[:min(len(self.words), 100)]
        for word in candidates:
            if word not in [g for g, _ in self.previous_guesses]:
                score = self._calculate_word_score(word)
                if score > best_score:
                    best_score = score
                    best_word = word
        return best_word or self.words[0]
    def _calculate_word_score(self, word):
        score = 0
        seen_letters = set()
        if word in [g for g, _ in self.previous_guesses]:
            return float('-inf')
        for i, letter in enumerate(word):
            if letter not in seen_letters:
                freq = self.letter_freq[letter] / len(self.words)
                score += freq * (1 - freq)  # Maximum info gain
                pos_freq = self.pos_freq[i][letter] / len(self.words)
                score += pos_freq * (1 - pos_freq)
                seen_letters.add(letter)
            else:
                if len(self.words) <= 5:
                    score *= 0.5
                else:
                    score *= 0.9
        return score
class MinimaxWordleSolver:
    def __init__(self, word_list):
        self.words = [w.upper() for w in word_list if len(w) == 5]
        self.previous_guesses = []
    def update_knowledge(self, guess, feedback):
        print(f"Minimax updating knowledge for guess: {guess} with feedback: {feedback}")
        old_count = len(self.words)
        self.previous_guesses.append((guess, feedback))
        filtered_words = self.words.copy()
        for prev_guess, prev_feedback in self.previous_guesses:
            filtered_words = [word for word in filtered_words
                              if self._matches_feedback(word, prev_guess, prev_feedback)]
        self.words = filtered_words
        print(f"Minimax filtered words from {old_count} to {len(self.words)}")
        if len(self.words) < 10:
            print(f"Minimax remaining words: {self.words}")
    def _matches_feedback(self, word, guess, feedback):
        word_letters = list(word)
        guess_letters = list(guess)
        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if f == "green" and word[i] != g:
                return False
            elif f == "green":
                word_letters[i] = '*'
                guess_letters[i] = '*'
        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if g == '*':
                continue
            if f == "yellow":
                if g not in word_letters:
                    return False
                word_letters[word_letters.index(g)] = '*'
            elif f == "gray" and g in word_letters:
                return False
        return True
    def get_best_guess(self, remaining_guesses):
        if not self.words:
            return None
        if len(self.words) <= 2:
            return self.words[0]
        best_score = float('-inf')
        best_word = None
        candidates = [w for w in self.words[:min(len(self.words), 50)] if len(w) == 5]  # Limit for performance
        for word in candidates:
            if word not in [g for g, _ in self.previous_guesses]:
                score = self._evaluate_guess(word, remaining_guesses)
                if score > best_score:
                    best_score = score
                    best_word = word
        return best_word or self.words[0]
    def _evaluate_guess(self, guess, remaining_guesses):
        patterns = self._get_possible_patterns(guess)
        worst_case_remaining = len(self.words)
        for pattern in patterns:
            remaining = sum(1 for word in self.words
                            if self._matches_feedback(word, guess, pattern))
            worst_case_remaining = min(worst_case_remaining,
                                       len(self.words) - remaining)
        if guess in self.words:
            worst_case_remaining += 0.1
        return worst_case_remaining
    def _get_possible_patterns(self, guess):
        patterns = set()
        for answer in self.words[:min(len(self.words), 20)]:
            pattern = []
            word_letters = list(answer)
            guess_letters = list(guess)
            # First pass: green letters
            for i, (g, w) in enumerate(zip(guess_letters, word_letters)):
                if g == w:
                    pattern.append("green")
                    word_letters[i] = '*'
                    guess_letters[i] = '*'
                else:
                    pattern.append("temp")
            for i, g in enumerate(guess_letters):
                if pattern[i] == "temp":
                    if g in word_letters:
                        pattern[i] = "yellow"
                        word_letters[word_letters.index(g)] = '*'
                    else:
                        pattern[i] = "gray"
            patterns.add(tuple(pattern))
        return patterns

class CSPWordleSolver:

    def __init__(self, word_list):
        self.words = [w.upper() for w in word_list if len(w) == 5]
        self.constraints = []

    def update_constraints(self, guess, feedback):
        self.constraints.append((guess, feedback))
        self._prune_domain()

    def _prune_domain(self):
        old_count = len(self.words)
        for guess, feedback in self.constraints:
            filtered_words = []
            for candidate in self.words:
                if self._matches_feedback(candidate, guess, feedback):
                    filtered_words.append(candidate)
            self.words = filtered_words
        print(f"CSP pruned words from {old_count} to {len(self.words)}")
        if len(self.words) < 10:
            print("CSP remaining words:", self.words)

    def _matches_feedback(self, candidate, guess, feedback):
        if len(candidate) != 5 or len(guess) != 5:
            return False

        cand_letters = list(candidate)
        guess_letters = list(guess)

        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if f == "green" and candidate[i] != g:
                return False
            if f == "green":
                cand_letters[i] = '*'
                guess_letters[i] = '*'

        for i, (g, f) in enumerate(zip(guess_letters, feedback)):
            if g == '*':
                continue
            if f == "yellow":
                if g not in cand_letters:
                    return False
                cand_letters[cand_letters.index(g)] = '*'
            elif f == "gray":
                if g in cand_letters:
                    return False
        return True

    def get_best_guess(self):
        if not self.words:
            return None
        if len(self.words) <= 2:
            return self.words[0]

        best_word = None
        best_score = -1
        for w in self.words[:min(len(self.words), 100)]:
            score = len(set(w))
            if score > best_score:
                best_score = score
                best_word = w
        return best_word or self.words[0]

def filter_words_based_on_feedback(guess, feedback, possible_words):
    filtered_words = []
    for word in possible_words:
        word = word.upper()
        valid = True
        used_positions = set()

        # Check green letters first
        for i, (letter, color) in enumerate(zip(guess, feedback)):
            if color == "green":
                if word[i] != letter:
                    valid = False
                    break
                used_positions.add(i)

        if not valid:
            continue

        # Check yellow letters
        for i, (letter, color) in enumerate(zip(guess, feedback)):
            if color == "yellow":
                if letter not in word or word[i] == letter:
                    valid = False
                    break

        if not valid:
            continue

        # Check gray letters
        for i, (letter, color) in enumerate(zip(guess, feedback)):
            if color == "gray":
                # Letter shouldn't exist in word unless it appears elsewhere as green/yellow
                letter_positions = [j for j in range(5) if guess[j] == letter]
                green_yellow_positions = [j for j in letter_positions if feedback[j] in ["green", "yellow"]]
                remaining_positions = [j for j in range(5) if j not in green_yellow_positions]

                if any(word[j] == letter for j in remaining_positions):
                    valid = False
                    break

        if valid:
            filtered_words.append(word)

    return filtered_words


def score_word(word, remaining_words):
    letter_freq = defaultdict(int)
    position_freq = defaultdict(lambda: defaultdict(int))

    for possible_word in remaining_words:
        for i, letter in enumerate(possible_word.upper()):
            letter_freq[letter] += 1
            position_freq[i][letter] += 1

    score = 0
    seen_letters = set()
    for i, letter in enumerate(word.upper()):
        if letter not in seen_letters:
            score += letter_freq[letter]
            score += position_freq[i][letter] * 2
            seen_letters.add(letter)

    return score




def handle_mouse_input(event):
    global cursor
    m_x, m_y = pygame.mouse.get_pos()
    for i, letter in enumerate(letters):
        x, y, ltr, color = letter
        if m_x > x - (LKEY_WIDTH if i == 27 else KEY_WIDTH)/2 and m_x < x + (LKEY_WIDTH if i == 27 else KEY_WIDTH)/2 and m_y > y - KEY_WIDTH/2 and m_y < y + KEY_WIDTH/2:
            if ltr == "<<":
                if cursor != 0:
                    cursor -= 1
                    boxes[5*guesses+cursor][2] = ""
            elif ltr == "Enter":
                if cursor == 5:
                    handle_guess()
            elif cursor < 5:
                boxes[5*guesses+cursor][2] = ltr
                cursor += 1

def handle_keyboard_input(event):
    global cursor
    if event.key == pygame.K_BACKSPACE:
        if cursor > 0:
            cursor -= 1
            boxes[5 * guesses + cursor][2] = ""
    elif event.key == pygame.K_RETURN:
        if cursor == 5:
            handle_guess()
    elif cursor < 5 and event.unicode.isalpha():
        boxes[5 * guesses + cursor][2] = event.unicode.upper()
        cursor += 1

def handle_guess():
    global run, cursor, guesses
    entered_word = ""
    for i in range(5):
        entered_word = entered_word + boxes[5*guesses+i][2]
    if entered_word.lower() not in allWords:
        display_message("not_in_list")
        return
    won = check_guess()
    draw()
    if won:
        print("You Won!")
        display_message("won")
        run = False
    elif guesses == 5:
        print("You Lost!")
        display_message("lost")
        run = False
    else:
        guesses += 1
        cursor = 0


def test_ai_solver(num_games=20):
    global ai_enabled, run, guesses, cursor, word
    ai_enabled = True

    results = {
        'wins': 0,
        'losses': 0,
        'avg_guesses': 0,
        'guesses_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        'failed_words': []
    }

    for game in range(num_games):
        setup_game()
        current_word = word
        game_won = False

        print(f"\nGame {game + 1}, Target word: {current_word}")

        while guesses < 6:
            current_guess = ai_guess()

            if not current_guess:
                print("Error: No valid guess available")
                break

            print(f"Guess {guesses + 1}: {current_guess}")

            for i, letter in enumerate(current_guess):
                boxes[5 * guesses + i][2] = letter
            cursor = 5

            if current_guess.lower() not in allWords:
                print(f"Invalid guess: {current_guess}")
                continue

            won = check_guess()
            draw()

            if won:
                results['wins'] += 1
                results['guesses_distribution'][guesses + 1] += 1
                results['avg_guesses'] += guesses + 1
                game_won = True
                print(f"Won in {guesses + 1} guesses!")
                break

            guesses += 1
            pygame.time.delay(500)

        if not game_won:
            results['losses'] += 1
            results['failed_words'].append(current_word)
            print(f"Lost! Word was {current_word}")

        pygame.time.delay(1000)

    if results['wins'] > 0:
        results['avg_guesses'] /= results['wins']

    print("\nTest Results:")
    print(f"Games played: {num_games}")
    print(f"Wins: {results['wins']} ({(results['wins'] / num_games) * 100:.1f}%)")
    print(f"Average guesses per win: {results['avg_guesses']:.2f}")
    print("\nGuess distribution:")
    for guesses_count, count in results['guesses_distribution'].items():
        if count > 0:
            print(f"{guesses_count} guesses: {count} times ({(count / results['wins']) * 100:.1f}% of wins)")
    print("\nFailed words:", results['failed_words'])

    return results


def test_csp_ai(num_games=20):
    global run, guesses, cursor, word

    results = {
        'wins': 0,
        'losses': 0,
        'avg_guesses': 0,
        'guesses_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        'failed_words': []
    }

    num_games = 10
    for game in range(num_games):
        setup_csp_game()
        current_word = word
        game_won = False

        print(f"\nCSP Game {game + 1}, Target word: {current_word}")

        while guesses < 6:
            current_guess = csp_ai_guess()

            if not current_guess:
                print("Error: No valid guess available")
                break

            print(f"CSP Guess {guesses + 1}: {current_guess}")

            for i, letter in enumerate(current_guess):
                boxes[5 * guesses + i][2] = letter
            cursor = 5

            if current_guess.lower() not in allWords:
                print(f"Invalid guess: {current_guess}")
                continue

            won = check_guess()
            draw()

            if won:
                results['wins'] += 1
                results['guesses_distribution'][guesses + 1] += 1
                results['avg_guesses'] += guesses + 1
                game_won = True
                print(f"CSP Won in {guesses + 1} guesses!")
                break

            guesses += 1
            pygame.time.delay(500)

        if not game_won:
            results['losses'] += 1
            results['failed_words'].append(current_word)
            print(f"CSP Lost! Word was {current_word}")

        pygame.time.delay(1000)

    if results['wins'] > 0:
        results['avg_guesses'] /= results['wins']

    print("\nCSP Test Results:")
    print(f"Games played: {num_games}")
    print(f"Wins: {results['wins']} ({(results['wins'] / num_games) * 100:.1f}%)")
    print(f"Average guesses per win: {results['avg_guesses']:.2f}")
    print("\nGuess distribution:")
    for guesses_count, count in results['guesses_distribution'].items():
        if count > 0:
            print(f"{guesses_count} guesses: {count} times ({(count / results['wins']) * 100:.1f}% of wins)")
    print("\nFailed words:", results['failed_words'])

    return results


def setup_csp_game():
    global run, ai_feedback, remaining_words, csp_solver
    run = True
    set_game_variables()
    load_text()
    pick_word()
    print("CSP Answer is: ", word)
    setup_keys()
    setup_boxes()
    ai_feedback = []
    remaining_words = wordleWords.copy()
    csp_solver = None


def csp_ai_guess():
    global ai_feedback, remaining_words, guesses, csp_solver

    print(f"\nCSP Making guess {guesses + 1}")

    if guesses == 0:
        csp_solver = CSPWordleSolver(wordleWords)
        return "STARE"

    current_guess = ""
    for i in range(5):
        current_guess += boxes[5 * (guesses - 1) + i][2]

    print(f"Updating CSP solver with guess: {current_guess} and feedback: {ai_feedback}")
    csp_solver.update_constraints(current_guess, ai_feedback)

    next_guess = csp_solver.get_best_guess()

    print(f"CSP solver remaining words: {len(csp_solver.words)}")
    print(f"CSP solver choosing word: {next_guess}")

    return next_guess


def astar_ai_guess():
    global ai_feedback, remaining_words, guesses, astar_solver

    print(f"\nA* Making guess {guesses + 1}")

    if guesses == 0:
        astar_solver = AStarWordleSolver(wordleWords)
        return "STARE"

    current_guess = ""
    for i in range(5):
        current_guess += boxes[5 * (guesses - 1) + i][2]

    print(f"Updating A* solver with guess: {current_guess} and feedback: {ai_feedback}")
    astar_solver.update_constraints(current_guess, ai_feedback)

    next_guess = astar_solver.get_best_guess()

    print(f"A* solver remaining words: {len(astar_solver.words)}")
    print(f"A* solver choosing word: {next_guess}")

    return next_guess


def setup_astar_game():
    global run, ai_feedback, remaining_words, astar_solver
    run = True
    set_game_variables()
    load_text()
    pick_word()
    print("A* Answer is: ", word)
    setup_keys()
    setup_boxes()
    ai_feedback = []
    remaining_words = wordleWords.copy()
    astar_solver = None


def test_astar_ai(num_games=20):
    global run, guesses, cursor, word

    results = {
        'wins': 0,
        'losses': 0,
        'avg_guesses': 0,
        'guesses_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        'failed_words': []
    }

    num_games = 10
    for game in range(num_games):
        setup_astar_game()
        current_word = word
        game_won = False

        print(f"\nA* Game {game + 1}, Target word: {current_word}")

        while guesses < 6:
            current_guess = astar_ai_guess()

            if not current_guess:
                print("Error: No valid guess available")
                break

            print(f"A* Guess {guesses + 1}: {current_guess}")

            for i, letter in enumerate(current_guess):
                boxes[5 * guesses + i][2] = letter
            cursor = 5

            if current_guess.lower() not in allWords:
                print(f"Invalid guess: {current_guess}")
                continue

            won = check_guess()
            draw()

            if won:
                results['wins'] += 1
                results['guesses_distribution'][guesses + 1] += 1
                results['avg_guesses'] += guesses + 1
                game_won = True
                print(f"A* Won in {guesses + 1} guesses!")
                break

            guesses += 1
            pygame.time.delay(500)

        if not game_won:
            results['losses'] += 1
            results['failed_words'].append(current_word)
            print(f"A* Lost! Word was {current_word}")

        pygame.time.delay(1000)

    if results['wins'] > 0:
        results['avg_guesses'] /= results['wins']

    print("\nA* Test Results:")
    print(f"Games played: {num_games}")
    print(f"Wins: {results['wins']} ({(results['wins'] / num_games) * 100:.1f}%)")
    print(f"Average guesses per win: {results['avg_guesses']:.2f}")
    print("\nGuess distribution:")
    for guesses_count, count in results['guesses_distribution'].items():
        if count > 0:
            print(f"{guesses_count} guesses: {count} times ({(count / results['wins']) * 100:.1f}% of wins)")
    print("\nFailed words:", results['failed_words'])

    return results

def print_results(solver_name, results):
    print(f"\n{solver_name} Test Results:")
    print(f"Games played: 10")
    print(f"Wins: {results['wins']} ({(results['wins'] / 10) * 100:.1f}%)")
    print(f"Average guesses per win: {results['avg_guesses']:.2f}")
    print("\nGuess distribution:")
    for guesses_count, count in results['guesses_distribution'].items():
        if count > 0:
            print(f"{guesses_count} guesses: {count} times ({(count / results['wins']) * 100:.1f}% of wins)")
    print("\nFailed words:", results['failed_words'])

def test_three_stage_hybrid(num_games=20):
    global run, guesses, cursor, word
    results = {
        'wins': 0, 'losses': 0, 'avg_guesses': 0,
        'guesses_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        'failed_words': []
    }

    num_games = 10
    for game in range(num_games):
        setup_three_stage_game()
        current_word = word
        game_won = False

        print(f"\nThree-Stage Game {game + 1}, Target word: {current_word}")

        while guesses < 6:
            current_guess = three_stage_hybrid_guess()
            if not current_guess:
                print("Error: No valid guess available")
                break

            print(f"Three-Stage Guess {guesses + 1}: {current_guess}")
            for i, letter in enumerate(current_guess):
                boxes[5 * guesses + i][2] = letter
            cursor = 5

            if current_guess.lower() not in allWords:
                print(f"Invalid guess: {current_guess}")
                continue

            won = check_guess()
            draw()

            if won:
                results['wins'] += 1
                results['guesses_distribution'][guesses + 1] += 1
                results['avg_guesses'] += guesses + 1
                game_won = True
                print(f"Three-Stage Won in {guesses + 1} guesses!")
                break

            guesses += 1
            pygame.time.delay(500)

        if not game_won:
            results['losses'] += 1
            results['failed_words'].append(current_word)

        pygame.time.delay(1000)

    if results['wins'] > 0:
        results['avg_guesses'] /= results['wins']

    print_results("Three-Stage Hybrid", results)
    return results


def csp_astar_hybrid_guess():
    global ai_feedback, remaining_words, guesses, csp_solver, astar_solver

    if guesses == 0:
        csp_solver = CSPWordleSolver(wordleWords)
        astar_solver = AStarWordleSolver(wordleWords)
        return "STARE"

    current_guess = ""
    for i in range(5):
        current_guess += boxes[5 * (guesses - 1) + i][2]
    csp_solver.update_constraints(current_guess, ai_feedback)
    astar_solver.update_constraints(current_guess, ai_feedback)

    if len(csp_solver.words) > 50:
        next_guess = astar_solver.get_best_guess()
        print(f"Using A* solver, {len(astar_solver.words)} words remain")
    else:
        next_guess = csp_solver.get_best_guess()
        print(f"Using CSP solver, {len(csp_solver.words)} words remain")

    return next_guess
def three_stage_hybrid_guess():
    global ai_feedback, remaining_words, guesses, bayesian_solver, astar_solver, minimax_solver

    if guesses == 0:
        bayesian_solver = BayesianWordleSolver(wordleWords)
        astar_solver = AStarWordleSolver(wordleWords)
        minimax_solver = MinimaxWordleSolver(wordleWords)
        return "STARE"

    current_guess = ""
    for i in range(5):
        current_guess += boxes[5 * (guesses - 1) + i][2]

    # Update all solvers
    bayesian_solver.update_beliefs(current_guess, ai_feedback)
    astar_solver.update_constraints(current_guess, ai_feedback)
    minimax_solver.update_knowledge(current_guess, ai_feedback)

    num_words = len(bayesian_solver.words)

    if num_words > 100:
        next_guess = bayesian_solver.get_best_guess()
        print(f"Using Bayesian solver, {num_words} words remain")
    elif num_words > 10:
        next_guess = astar_solver.get_best_guess()
        print(f"Using A* solver, {num_words} words remain")
    else:
        next_guess = minimax_solver.get_best_guess(6 - guesses)
        print(f"Using Minimax solver, {num_words} words remain")

    return next_guess
def csp_bayesian_hybrid_guess():
    global ai_feedback, remaining_words, guesses, csp_solver, bayesian_solver

    if guesses == 0:
        csp_solver = CSPWordleSolver(wordleWords)
        bayesian_solver = BayesianWordleSolver(wordleWords)
        return "STARE"

    current_guess = ""
    for i in range(5):
        current_guess += boxes[5 * (guesses - 1) + i][2]

    csp_solver.update_constraints(current_guess, ai_feedback)
    bayesian_solver.update_beliefs(current_guess, ai_feedback)

    if len(csp_solver.words) > 20:
        next_guess = bayesian_solver.get_best_guess()
        print(f"Using Bayesian solver, {len(bayesian_solver.words)} words remain")
    else:
        next_guess = csp_solver.get_best_guess()
        print(f"Using CSP solver, {len(csp_solver.words)} words remain")

    return next_guess
def setup_csp_astar_game():
    global run, ai_feedback, remaining_words, csp_solver, astar_solver
    run = True
    set_game_variables()
    load_text()
    pick_word()
    print("CSP-A* Answer is: ", word)
    setup_keys()
    setup_boxes()
    ai_feedback = []
    remaining_words = wordleWords.copy()
    csp_solver = None
    astar_solver = None

def setup_three_stage_game():
    global run, ai_feedback, remaining_words, bayesian_solver, astar_solver, minimax_solver
    run = True
    set_game_variables()
    load_text()
    pick_word()
    print("Three-Stage Answer is: ", word)
    setup_keys()
    setup_boxes()
    ai_feedback = []
    remaining_words = wordleWords.copy()
    bayesian_solver = None
    astar_solver = None
    minimax_solver = None

def setup_csp_bayesian_game():
    global run, ai_feedback, remaining_words, csp_solver, bayesian_solver
    run = True
    set_game_variables()
    load_text()
    pick_word()
    print("CSP-Bayesian Answer is: ", word)
    setup_keys()
    setup_boxes()
    ai_feedback = []
    remaining_words = wordleWords.copy()
    csp_solver = None
    bayesian_solver = None
def test_csp_astar_hybrid(num_games=20):
    global run, guesses, cursor, word
    results = {
        'wins': 0, 'losses': 0, 'avg_guesses': 0,
        'guesses_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        'failed_words': []
    }

    num_games = 10
    for game in range(num_games):
        setup_csp_astar_game()
        current_word = word
        game_won = False

        print(f"\nCSP-A* Game {game + 1}, Target word: {current_word}")

        while guesses < 6:
            current_guess = csp_astar_hybrid_guess()
            if not current_guess:
                print("Error: No valid guess available")
                break

            print(f"CSP-A* Guess {guesses + 1}: {current_guess}")
            for i, letter in enumerate(current_guess):
                boxes[5 * guesses + i][2] = letter
            cursor = 5

            if current_guess.lower() not in allWords:
                print(f"Invalid guess: {current_guess}")
                continue

            won = check_guess()
            draw()

            if won:
                results['wins'] += 1
                results['guesses_distribution'][guesses + 1] += 1
                results['avg_guesses'] += guesses + 1
                game_won = True
                print(f"CSP-A* Won in {guesses + 1} guesses!")
                break

            guesses += 1
            pygame.time.delay(500)

        if not game_won:
            results['losses'] += 1
            results['failed_words'].append(current_word)

        pygame.time.delay(1000)

    if results['wins'] > 0:
        results['avg_guesses'] /= results['wins']

    print_results("CSP-A* Hybrid", results)
    return results
def test_csp_bayesian_hybrid(num_games=20):
    global run, guesses, cursor, word
    results = {
        'wins': 0, 'losses': 0, 'avg_guesses': 0,
        'guesses_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0},
        'failed_words': []
    }

    num_games = 10
    for game in range(num_games):
        setup_csp_bayesian_game()
        current_word = word
        game_won = False

        print(f"\nCSP-Bayesian Game {game + 1}, Target word: {current_word}")

        while guesses < 6:
            current_guess = csp_bayesian_hybrid_guess()
            if not current_guess:
                print("Error: No valid guess available")
                break

            print(f"CSP-Bayesian Guess {guesses + 1}: {current_guess}")
            for i, letter in enumerate(current_guess):
                boxes[5 * guesses + i][2] = letter
            cursor = 5

            if current_guess.lower() not in allWords:
                print(f"Invalid guess: {current_guess}")
                continue

            won = check_guess()
            draw()

            if won:
                results['wins'] += 1
                results['guesses_distribution'][guesses + 1] += 1
                results['avg_guesses'] += guesses + 1
                game_won = True
                print(f"CSP-Bayesian Won in {guesses + 1} guesses!")
                break

            guesses += 1
            pygame.time.delay(500)

        if not game_won:
            results['losses'] += 1
            results['failed_words'].append(current_word)

        pygame.time.delay(1000)

    if results['wins'] > 0:
        results['avg_guesses'] /= results['wins']

    print_results("CSP-Bayesian Hybrid", results)
    return results
def main():
    global run
    if ai_enabled:
        print("Running original hybrid solver test:")
        hybrid_results = test_ai_solver(10)
        print("\n" + "=" * 50 + "\n")

        print("Running CSP solver test:")
        csp_results = test_csp_ai()
        print("\n" + "=" * 50 + "\n")

        print("Running A* solver test:")
        astar_results = test_astar_ai()
        print("\n" + "=" * 50 + "\n")

        print("Running CSP-A* hybrid test:")
        csp_astar_results = test_csp_astar_hybrid()
        print("\n" + "=" * 50 + "\n")

        print("Running Three-Stage hybrid test:")
        three_stage_results = test_three_stage_hybrid()
        print("\n" + "=" * 50 + "\n")

        print("Running CSP-Bayesian hybrid test:")
        csp_bayesian_results = test_csp_bayesian_hybrid()

        print("\n" + "=" * 50)
        print("COMPARATIVE RESULTS:")
        print("=" * 50)

        solvers = {
            "Original Hybrid": hybrid_results,
            "CSP": csp_results,
            "A*": astar_results,
            "CSP-A* Hybrid": csp_astar_results,
            "Three-Stage Hybrid": three_stage_results,
            "CSP-Bayesian Hybrid": csp_bayesian_results
        }

        for name, results in solvers.items():
            print(f"\n{name}:")
            print(f"Win Rate: {(results['wins'] / 10) * 100:.1f}%")
            print(f"Average Guesses: {results['avg_guesses']:.2f}")
            if results['failed_words']:
                print(f"Failed words: {results['failed_words']}")
        strategies = []
        avg_guesses = []
        win_rate = []
        guess_distribution = []
        losses = []
        for name, result in solvers.items():
            strategies.append(name)
            avg_guesses.append(result['avg_guesses'])  
            win_rate.append(result['wins'])
            guess_distribution.append(result['guesses_distribution'])
            losses.append(result['losses'])
        x = np.arange(len(strategies))
        fig, ax1 = plt.subplots(figsize=(10, 6))
        bar_width = 0.35
        ax1.set_ylim(0, 6)

        bar1 = ax1.bar(x - bar_width / 2, avg_guesses, bar_width, label='Avg Guesses', color='b', alpha=0.7)
        ax2 = ax1.twinx()
        bar2 = ax2.bar(x + bar_width / 2, win_rate, bar_width, label='Win Rate (%)', color='g', alpha=0.7)
    
        ax1.set_xlabel('Strategies')
        ax1.set_ylabel('Average Guesses', color='b')
        ax2.set_ylabel('Win Rate (%)', color='g')
        ax1.set_title('Comparison of Wordle Strategies')
        ax1.set_xticks(x)
        ax1.set_xticklabels(strategies)

        bars = bar1 + bar2
        labels = [bar.get_label() for bar in [bar1, bar2]]
        plt.legend(bars, labels, loc='upper left')

        plt.tight_layout()
        plt.show()
        for i, distribution in enumerate(guess_distribution):
            plt.figure(figsize=(8, 5))
            guesses = list(distribution.keys())
            frequencies = list(distribution.values())         
            plt.bar(guesses, frequencies, color='skyblue', alpha=0.7)
            plt.title(f'Guess Distribution for {strategies[i]}', fontsize=14)
            plt.xlabel('Number of Guesses', fontsize=12)
            plt.ylabel('Frequency', fontsize=12)
            plt.xticks(guesses)
            plt.grid(axis='y', linestyle='--', alpha=0.6)
            plt.tight_layout()
            plt.show()
    else:
        setup_game()
        while run:
            clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    pygame.quit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    handle_mouse_input(event)
                elif event.type == pygame.KEYDOWN:
                    handle_keyboard_input(event)
            draw()
        main()

if __name__ == "__main__":
    main()