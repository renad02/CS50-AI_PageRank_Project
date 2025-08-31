import os                                                                       # For reading files and directories
import random                                                                   # For random sampling in sample_pagerank
import re                                                                       # For extracting links from HTML using regex
import sys                                                                      # For command-line arguments

DAMPING = 0.85                                                                  # Probability of following a link vs random jump
SAMPLES = 10000                                                                 # Number of samples for the sampling method


def main():
    # Ensure user provides the corpus directory
    if len(sys.argv) != 2:
        sys.exit("Usage: python pagerank.py corpus")
        
    # Load the corpus (all HTML files and their links)
    corpus = crawl(sys.argv[1])
    
    # --- Sampling method ---
    ranks = sample_pagerank(corpus, DAMPING, SAMPLES)
    print(f"PageRank Results from Sampling (n = {SAMPLES})")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")
        
    # --- Iterative method ---
    ranks = iterate_pagerank(corpus, DAMPING)
    print(f"PageRank Results from Iteration")
    for page in sorted(ranks):
        print(f"  {page}: {ranks[page]:.4f}")


def crawl(directory):                                                           # function that builds the corpus graph from HTML files.
    """
    Parse a directory of HTML pages and check for links to other pages.
    Return a dictionary where each key is a page, and values are
    a list of all other pages in the corpus that are linked to by the page.
    """
    pages = dict()                                                              # Initialize mapping: filename → set of outgoing links (only to files in corpus).

    # Extract all links from HTML files
    for filename in os.listdir(directory):
        if not filename.endswith(".html"):
            continue
        with open(os.path.join(directory, filename)) as f:                      # Open the HTML file safely.
            contents = f.read()                                                 # Read entire file content into a string.
            links = re.findall(r"<a\s+(?:[^>]*?)href=\"([^\"]*)\"", contents)   # se regex to capture all double‑quoted href attribute values in <a ...> tags. Breakdown: \s+ allows whitespace after <a; (?:[^>]*?) non‑greedy match of other attributes; href=\"([^\"]*)\" captures the URL inside quotes.
            pages[filename] = set(links) - {filename}                           # Convert to a set (unique targets) and remove self‑links to avoid self‑edges.

    # Only include links to other pages in the corpus
    for filename in pages:
        pages[filename] = set(
            link for link in pages[filename]
            if link in pages
        )

    return pages                                                                # Return the final corpus graph (dict of page → set(outgoing links)).


def transition_model(corpus, page, damping_factor):                             # Function to build P(next‑page) given current page.
    """
    Return a probability distribution over which page to visit next,
    given a current page.

    With probability `damping_factor`, choose a link at random
    linked to by `page`. With probability `1 - damping_factor`, choose
    a link at random chosen from all pages in the corpus.
    """
    distribution = dict()                                                       # Start an empty dict for the probability distribution over pages.
    n = len(corpus)                                                             # Number of pages in the corpus.
    links = corpus[page]                                                        # The set of outgoing links from the current page.

    if links:                                                                   # Case 1: current page has one or more outgoing links.
        for p in corpus:                                                        # Loop through every page in the entire corpus. We’re about to initialize a probability distribution for the random surfer model.
            distribution[p] = (1 - damping_factor) / n                          # Assign random‑jump mass equally to all pages. (1 - damping_factor) is the probability of a random jump (not following links).
                                                                                # / n spreads that random-jump probability equally among all n pages in the corpus.
                                                                                # If damping_factor = 0.85, then 1 - damping_factor = 0.15.
                                                                                # That 15% chance is split evenly across all pages (like “teleportation”).
                                                                                
        for linked_page in links:                                               # Now we loop only over the outgoing links of the current page. These are the pages that the random surfer could actually click on from the current page.
            distribution[linked_page] += damping_factor / len(links)
    else:                                                                       # Case 2: dangling page (no outgoing links).
        for p in corpus:                                                        # We add the “link-following” probability to each outgoing neighbor.
            distribution[p] = 1 / n                                             # - damping_factor is the probability of following a link (usually 0.85 → 85%).
                                                                                # - len(links) is the number of outgoing links.
                                                                                # So each linked page gets an equal share of this 85%.
    return distribution                                                         # Return the full categorical distribution P(next page | current page).


def sample_pagerank(corpus, damping_factor, n):                                 # estimate how important each page is by simulating a person randomly surfing the web thousands of times.
    """
    Return PageRank values for each page by sampling `n` pages
    according to transition model, starting with a page at random.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """
    distribution = {}                                                           # Initialize a dictionary to count estimated probabilities.
    for page in corpus:
        distribution[page] = 0                                                  # Start with zero visits for every page.
    
    page = random.choice(list(corpus.keys()))                                   # Pick a random page as the starting point of the surfer. The surfer has to start somewhere.

    for i in range(1, n):                                                       # Repeat the simulation n - 1 times. Each loop = one random “click” / step in the browsing process.
        current_distribution = transition_model(corpus, page, damping_factor)   # Use the transition_model to compute the probability distribution for the next page, given the current one.
        for page in distribution:                                               # Update the estimated PageRank values. This way, we don’t store all samples — we just keep a running average.
            distribution[page] = ((i-1) * distribution[page] + current_distribution[page]) / i  # Old average weighted by (i-1). Add new observation (current_distribution[page]). Divide by total samples i. 
        
        page = random.choices(list(distribution.keys()), list(distribution.values()), k=1)[0]   # Pick the next page to move to, using the probability distribution we just computed.random.choices selects one page according to the weights. [0] extracts the single choice from the list.

    return distribution                                                         # After n samples, return the estimated PageRank values. All values add up to ~1 (since they’re probabilities).


def iterate_pagerank(corpus, damping_factor):
    """
    Return PageRank values for each page by iteratively updating
    PageRank values until convergence.

    Return a dictionary where keys are page names, and values are
    their estimated PageRank value (a value between 0 and 1). All
    PageRank values should sum to 1.
    """ 
    ranks = {}                                                                  # dictionary to hold the PageRank score of each page.
    threshold = 0.0005                                                          # We stop updating once the rank of each page changes less than 0.001 between two iterations.
    N = len(corpus)                                                             # number of pages in the corpus.
    
    for key in corpus:                                                          # At first, we assume all pages are equally important.
        ranks[key] = 1 / N                                                      # Example: 4 pages → each starts with rank 0.25.


    while True:
        count = 0                                                               # keeps track of how many pages have converged in this round.
        for key in corpus:                                                      # For a given key (the page we’re calculating rank for):
            new = (1 - damping_factor) / N                                      # Start with random jump probability (1 - d)/N.
            sigma = 0                                                           # Initialize sigma = 0 (this will accumulate contributions from other pages).
            for page in corpus:                                                 # Loop over all pages (page).
                if key in corpus[page]:                                         
                    num_links = len(corpus[page])                               # If page has a hyperlink pointing to key, then it contributes part of its rank.
                    sigma = sigma + ranks[page] / num_links                     # Contribution formula = (rank of page ÷ number of links page has).
                                                                                # If page A has rank 0.4 and 2 outgoing links, then each linked page gets 0.2.
            sigma = damping_factor * sigma                                      # Scale contributions by damping factor d. Add this to the base random jump.
            new += sigma                                                        # Now new is the updated rank of the current page.
            if abs(ranks[key] - new) < threshold:                               # Compare the old rank vs the new rank.
                count += 1                                                      # If the difference is smaller than threshold, this page has converged.
            ranks[key] = new                                                    # Regardless, update the page’s rank to the new value.
        if count == N:                                                          # If all N pages converged in this round, stop the loop. Otherwise, keep iterating.
            break
    return ranks                                                                # Once loop breaks, the dictionary ranks contains the final PageRank values.


if __name__ == "__main__":
    main()
