# Import necessary libraries
import re
import random
import time

# Part 1: Advanced Regular Expressions (Regex)

# Task 1: Extract URLs
def extract_urls(text):
    """
    Task: Write a regex pattern to extract all URLs from the given text.
    Input:
        text (str): The input text containing URLs.
    Output:
        A list of extracted URLs.
    """
    # Write your regex pattern here
    urls = re.findall(r'https?://[^\s]+', text)
    return urls

# Task 2: Extract balanced parentheses
def extract_balanced_parentheses(text):
    """
    Task: Write a regex pattern to extract all balanced parentheses from the given text.
    Input:
        text (str): The input text containing parentheses.
    Output:
        A list of balanced parentheses.
    """
    # Write your regex pattern here
    p1 = re.findall(r'\([\S*]\)', text)
    p2 = re.findall(r'\([\S*]\(?[\S*]\)?[\S*]\)', text)
    p1.extend(p2)
    return p1

def generate_phone_number():
    area_code = random.randint(100, 999)
    prefix = random.randint(100, 999)
    local = random.randint(1000, 9999)
    return f"({area_code})-{prefix}-{local}"

# Task 3: Compare regex patterns for phone numbers
def compare_phone_number_patterns(phone_numbers):
    """
    Task: Compare the efficiency of two regex patterns for validating phone numbers.
    Input:
        phone_numbers (list): A list of phone numbers to test.
    Output:
        Performance results for each regex pattern.
    """
    pattern1 = r"^\(\d{3}\) \d{3}-\d{4}$"  # Example regex pattern 1
    pattern2 = r"^[(]{1}\d{3}[)]{1} \d{3}-\d{4}$"  # Example regex pattern 2
    
    # Implement your testing logic here
    data = [generate_phone_number() for _ in range(10000)]
    pattern1 = re.compile(pattern1)
    pattern2 = re.compile(pattern2)

    st1 = time.time()
    for d in data:
        pattern1.match(d)
    et1 = time.time()

    st2 = time.time()
    for d in data:
        pattern2.match(d)
    et2 = time.time()

    p1_time = et1 - st1
    p2_time = et2 - st2
    print(f"computational time for pattern 1: {p1_time}ms")
    print(f"computational time for pattern 2: {p2_time}ms")
    if p1_time > p2_time:
        print("pattern 2 is computationally efficient")
    else:
        print("pattern 1 is computationally efficient")


# Part 2: Edit Distance – Weighted and Practical Applications

# Task 1: Weighted Edit Distance
def weighted_edit_distance(str1, str2):
    """
    Task: Implement the weighted edit distance algorithm.
    Weights:
        - Insertion = 1
        - Deletion = 2
        - Substitution = 3
    Input:
        str1 (str): First input string.
        str2 (str): Second input string.
    Output:
        An integer representing the weighted edit distance.
    """
    # Initialize the DP table
    str1 = "#" + str1
    str2 = "#" + str2
    n_rows = len(str1)
    n_cols = len(str2)
    dp_table = [[0 for _ in range(n_cols)] for _ in range(n_rows)]

    # Fill the DP table
    for r in (range(n_rows)):
        for c in (range(n_cols)):
            # if r>0:
            #     print("c:", c)
            if r == 0:
                dp_table[r][c] = c
            elif c == 0:
                dp_table[r][c] = r
            else:
                dp_table[r][c] = min(
                    dp_table[r-1][c] + 1,
                    dp_table[r][c-1] + 1,
                    dp_table[r-1][c-1] + (2 if str1[r] != str2[c] else 0)
                )

    # Return the weighted edit distance
    return dp_table[-1][-1]

# Task 2: Closest match using edit distance
def closest_match(word, dataset):
    """
    Task: Find the closest match to a given word in a dataset using edit distance.
    Input:
        word (str): The input word.
        dataset (list): A list of words.
    Output:
        The closest matching word.
    """
    # Write your logic here
    edit_distances = [weighted_edit_distance(word, d) for d in dataset]
    match_index = edit_distances.index(min(edit_distances))
    return dataset[match_index]

# Task 3: Print DP table for Levenshtein distance
def print_dp_table(str1, str2):
    """
    Task: Implement the Levenshtein distance algorithm and print the DP table.
    Input:
        str1 (str): First input string.
        str2 (str): Second input string.
    Output:
        Print the DP table.
    """
    # Initialize and fill the DP table
    str1 = "#" + str1
    str2 = "#" + str2
    n_rows = len(str1)
    n_cols = len(str2)
    dp_table = [[0 for _ in range(n_cols)] for _ in range(n_rows)]

    # Fill the DP table
    for r in (range(n_rows)):
        for c in (range(n_cols)):
            # if r>0:
            #     print("c:", c)
            if r == 0:
                dp_table[r][c] = c
            elif c == 0:
                dp_table[r][c] = r
            else:
                dp_table[r][c] = min(
                    dp_table[r-1][c] + 1,
                    dp_table[r][c-1] + 1,
                    dp_table[r-1][c-1] + (1 if str1[r] != str2[c] else 0)
                )


    # Print the DP table
    for row in range(len(dp_table)-1, -1, -1):
        print(f"{str1[row]} {dp_table[row]}")
    str2 = ", ".join(str2)
    print(f"   {str2}")

# Example usage (uncomment to test your functions)
# print("Extracted URLs:", extract_urls("Visit us at https://www.example.com or http://blog.example.org/index.html"))
# print(extract_balanced_parentheses("a(b)c(d)e(f(g)h)i")) 
# print("Weighted Edit Distance:", weighted_edit_distance("data", "date"))
# print("Closest Match:", closest_match("natrual", ["natural", "language", "processing", "data", "science"]))
# print_dp_table("kitten", "sitting")