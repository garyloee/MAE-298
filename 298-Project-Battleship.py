import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
#random.seed(24567453)
random.seed(399)
def generate_ship():
    direction = random.choice(['horizontal', 'vertical'])
    if direction == 'horizontal':
        row = random.randint(0, 4)
        col = random.randint(0, 2)
        ship_coords = [(row, col + i) for i in range(3)]
    else:
        row = random.randint(0, 2)
        col = random.randint(0, 4)
        ship_coords = [(row + i, col) for i in range(3)]
    return ship_coords

ship = generate_ship()
print("Ship's Position:", ship)

def initialize_prior():
    prior = np.array([
        [1.2, 1.3, 1.4, 1.3, 1.2],
        [1.3, 1.4, 1.5, 1.4, 1.3],
        [1.4, 1.5, 1.6, 1.5, 1.4],
        [1.3, 1.4, 1.5, 1.4, 1.3],
        [1.2, 1.3, 1.4, 1.3, 1.2]
    ])

    prior /= prior.sum()  # Normalize to maintain it as a probability distribution
    return prior

prior = initialize_prior()
print("Initial Prior Probability Map:\n", prior)

def update_probability_map(prior, guess, hit):
    row, col = guess
    if hit:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < 5 and 0 <= c < 5:
                prior[r, c] *= 2  # Increase the probability for adjacent cells
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            r, c = row + dr, col + dc
            if 0 <= r < 5 and 0 <= c < 5:
                prior[r, c] *= 1.75  # Increase the probability for adjacent cells
        prior[row, col] = 0  # We already know this cell is part of the ship
    else:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            r, c = row + dr, col + dc
            if 0 <= r < 5 and 0 <= c < 5:
                prior[r, c] *= 0.8  # decrease the probability for adjacent cells
        for dr, dc in [(-2, 0), (2, 0), (0, -2), (0, 2)]:
            r, c = row + dr, col + dc
            if 0 <= r < 5 and 0 <= c < 5:
                prior[r, c] *= 0.9  # decrease the probability for adjacent cells
        prior[row, col] = 0  # No ship here
    prior /= prior.sum()  # Normalize to maintain it as a probability distribution
    return prior

# Simulate the search process
def search_ship(prior, ship):
    guesses = []
    while True:
        guess = np.unravel_index(np.argmax(prior, axis=None), prior.shape)
        guesses.append(guess)
        hit = guess in ship
        prior = update_probability_map(prior, guess, hit)
        if all(cell in guesses for cell in ship):
            break
    return guesses, prior

guesses, final_prior = search_ship(prior, ship)
print("Final Probability Map After Search:\n", final_prior)
print("Guesses Made:", guesses)



def plot_probability_map(prob_map, title):
    plt.figure(figsize=(6, 6))
    sns.heatmap(prob_map, annot=True, cmap="YlGnBu", cbar=False, square=True)
    plt.title(title)
    plt.show()

# Simulate and plot at each step
prior = initialize_prior()
plot_probability_map(prior, "Initial Prior Probability Map")

for guess in guesses:
    hit = guess in ship
    prior = update_probability_map(prior, guess, hit)
    plot_probability_map(prior, f"After Guess {guess} - {'Hit' if hit else 'Miss'}")


print("Final Ship Position:\n", ship)
print("Number of Guesses:",len(guesses))

def plot_final_map(ship):
    # Create a 5x5 grid initialized with zeros
    grid = np.zeros((5, 5))
    
    # Mark the ship's position with 1s
    for (r, c) in ship:
        grid[r, c] = 1

    plt.figure(figsize=(6, 6))
    sns.heatmap(grid, annot=True, cmap="Blues", cbar=False, square=True, 
                linewidths=0.5, linecolor='black')
    plt.title("Final Ship Position")
    plt.show()

# Assuming the 'ship' variable has been defined earlier
plot_final_map(ship)