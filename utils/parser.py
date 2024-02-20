import re
from langchain.schema import BaseOutputParser

class MyOutputParser(BaseOutputParser):
    def parse(self, text):
        # Define the patterns for extracting strings enclosed in single or double quotes
        patterns = [r"'(.*?)'", r'"(.*?)"']

        matches = []
        for pattern in patterns:
            # Use re.findall to find all matches of the pattern in the input string
            matches.extend(re.findall(pattern, text))

        return matches

def find_max_words(keywords):
    # Initialize a variable to store the maximum number of words
    max_words = 0

    # Iterate through each string in the list
    for string in keywords:
        # Split the string into words using the split() method
        words = string.split()
        
        # Update max_words if the current string has more words
        max_words = max(max_words, len(words))
    return max_words

def get_context_length(error_message):
    if "Trial key, which is limited to 5 API calls / minute" in error_message:
        raise Exception("API Rate Limit")
    # Use regular expression to find all numbers in the string
    numbers = re.findall(r'\d+', error_message)

    # Convert the list of strings to a list of integers
    numbers = list(map(int, numbers))

    return numbers[0], numbers[1]