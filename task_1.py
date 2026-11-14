"""Порівняння алгоритмів пошуку підрядка.

Програма читає локальний файл README.md, у якому містяться тексти двох
статей. Із README.md вона виділяє статтю 1 і статтю 2, а потім шукає в них
два види підрядків: підрядок, який є в тексті, та вигаданий підрядок,
якого в тексті немає.

Для пошуку використовуються алгоритми Боєра-Мура, Кнута-Морріса-Пратта
та Рабіна-Карпа. Модуль timeit вимірює час роботи кожного алгоритму.
Після цього програма визначає найшвидший алгоритм для кожної статті
та для всіх тестів разом.

Для статті 1 шукаються підрядки
«фундаментальними знаннями» та
«вигаданий_підрядок_якого_немає_в_жодній_статті».
Для статті 2 шукаються підрядки
«Профілювання варіанту із інвертованим списком» та
«вигаданий_підрядок_якого_немає_в_жодній_статті».

Після запуску на екрані можна побачити назви прочитаних статей, вибрані
підрядки, час виконання кожного алгоритму та повідомлення про найшвидший
алгоритм. Це потрібно для практичного порівняння ефективності алгоритмів
на однакових текстах і підрядках та для визначення алгоритму, який найкраще
працює в умовах цього експерименту.

Запуск програми з командного рядка:
python3 task_1.py
"""

# Import the module used to measure execution time.
import timeit
# Import the type used to describe functions passed to the benchmark.
from collections.abc import Callable
# Import Path for safe work with file system paths.
from pathlib import Path


def boyer_moore(text: str, pattern: str) -> int:
    """Return the first occurrence of pattern using the Boyer-Moore algorithm. Text is a string that contains the text
    to be searched, and pattern is a string that contains the substring to be found. The function returns the index of
    the first occurrence of pattern in text, or -1 if pattern is not found."""
    # An empty pattern is considered to be found at the beginning of the text.
    if not pattern:
        return 0

    # Store the lengths of the pattern and the text.
    pattern_length = len(pattern)
    text_length = len(text)
    # A longer pattern cannot be found in a shorter text.
    if pattern_length > text_length:
        return -1

    # Store the last position of each character to calculate safe shifts.
    last_occurrence = {
        character: index for index, character in enumerate(pattern)
    }
    # Start comparing the pattern with the first text window.
    shift = 0

    # Continue while the pattern can still fit into the remaining text.
    while shift <= text_length - pattern_length:
        # Start comparison with the last character of the pattern.
        pattern_index = pattern_length - 1

        # Compare characters from right to left until a mismatch is found.
        while (
            pattern_index >= 0
            and pattern[pattern_index] == text[shift + pattern_index]
        ):
            # Move left while the compared characters are equal.
            pattern_index -= 1

        # A negative index means that the entire pattern was found.
        if pattern_index < 0:
            return shift

        # Read the text character that caused the mismatch.
        mismatched_character = text[shift + pattern_index]
        # Shift the pattern to skip positions that cannot match.
        shift += max(
            1,
            pattern_index - last_occurrence.get(mismatched_character, -1),
        )

    return -1


def knuth_morris_pratt(text: str, pattern: str) -> int:
    """Return the first occurrence of pattern using the Knuth-Morris-Pratt algorithm."""
    # An empty pattern is considered to be found at the beginning of the text.
    if not pattern:
        return 0

    # Build the longest-prefix-suffix table used to skip repeated comparisons.
    prefix_lengths = [0] * len(pattern)
    # Set indexes for building the prefix table.
    prefix_index = 0
    pattern_index = 1

    # Build the table until every pattern character is processed.
    while pattern_index < len(pattern):
        if pattern[pattern_index] == pattern[prefix_index]:
            # Extend the current matching prefix.
            prefix_index += 1
            prefix_lengths[pattern_index] = prefix_index
            pattern_index += 1
        elif prefix_index:
            # Use a previously calculated prefix after a mismatch.
            prefix_index = prefix_lengths[prefix_index - 1]
        else:
            # Move to the next pattern character when no prefix exists.
            pattern_index += 1

    # Start scanning the text and the pattern from their first characters.
    text_index = 0
    pattern_index = 0

    # Reuse known prefixes after a mismatch instead of restarting the search.
    while text_index < len(text):
        if text[text_index] == pattern[pattern_index]:
            # Move both indexes after matching characters.
            text_index += 1
            pattern_index += 1
            if pattern_index == len(pattern):
                # Return the beginning of the complete matching substring.
                return text_index - pattern_index
        elif pattern_index:
            # Jump to the longest suitable prefix after a mismatch.
            pattern_index = prefix_lengths[pattern_index - 1]
        else:
            # Move to the next text character when no prefix can be reused.
            text_index += 1

    # Return -1 when the pattern was not found in the text.
    return -1


def rabin_karp(text: str, pattern: str) -> int:
    """Return the first occurrence of pattern using the Rabin-Karp algorithm."""
    # An empty pattern is considered to be found at the beginning of the text.
    if not pattern:
        return 0

    # Store the lengths of the pattern and the text.
    pattern_length = len(pattern)
    text_length = len(text)
    # A longer pattern cannot be found in a shorter text.
    if pattern_length > text_length:
        return -1

    # A rolling hash narrows comparisons; matching hashes are checked character by character.
    base = 256
    modulus = 1_000_000_007
    highest_power = pow(base, pattern_length - 1, modulus)
    pattern_hash = 0
    window_hash = 0

    # Calculate hashes for the pattern and the first text window.
    for index in range(pattern_length):
        pattern_hash = (pattern_hash * base + ord(pattern[index])) % modulus
        window_hash = (window_hash * base + ord(text[index])) % modulus

    # Move the text window one character at a time.
    for start in range(text_length - pattern_length + 1):
        # Compare actual strings when their hashes match.
        if pattern_hash == window_hash and (
            text[start : start + pattern_length] == pattern
        ):
            # Return the start position of the matching substring.
            return start

        if start < text_length - pattern_length:
            # Remove the old character and add the next one to the rolling hash.
            window_hash = (
                (window_hash - ord(text[start]) * highest_power) * base
                + ord(text[start + pattern_length])
            ) % modulus

    # Return -1 when no matching window was found.
    return -1


def load_articles(readme_path: Path) -> dict[str, str]:
    """Read README.md and extract the two article sections from it."""
    # Read the local README file because it contains the article texts used for the search.
    readme = readme_path.read_text(encoding="utf-8")
    # Search for markers that separate article 1 from article 2.
    article_one_marker = "стаття 1:"
    article_two_marker = "стаття 2:"

    # Stop with an explicit error if either article marker is missing.
    if article_one_marker not in readme or article_two_marker not in readme:
        raise ValueError(
            f"Could not find both article sections in {readme_path}."
        )

    # Extract the text after the first article marker.
    article_one = readme.split(article_one_marker, maxsplit=1)[1]
    # Split the remaining text into article 1 and article 2.
    article_one, article_two = article_one.split(article_two_marker, maxsplit=1)
    # Return both cleaned article texts in a dictionary.
    return {
        "Article 1": article_one.strip(),
        "Article 2": article_two.strip(),
    }


def benchmark(
    algorithm: Callable[[str, str], int],
    text: str,
    pattern: str,
    repetitions: int = 5,
    number: int = 1_000,
) -> float:
    """Return the best average execution time for one search in seconds."""
    # Create a timer that calls the selected search algorithm.
    timer = timeit.Timer(lambda: algorithm(text, pattern))
    # Repeat the measurement and return the best average result.
    return min(timer.repeat(repeat=repetitions, number=number)) / number


def main() -> None:
    """Benchmark all algorithms for an existing and a missing substring."""
    # Build the path to README.md located next to this script.
    readme_path = Path(__file__).with_name("README.md")
    # Read and separate the two articles from README.md.
    articles = load_articles(readme_path)
    # Store the algorithms that will be compared.
    algorithms: dict[str, Callable[[str, str], int]] = {
        "Boyer-Moore": boyer_moore,
        "Knuth-Morris-Pratt": knuth_morris_pratt,
        "Rabin-Karp": rabin_karp,
    }
    # Store one existing and one missing pattern for each article.
    patterns = {
        "Article 1": {
            "existing": "фундаментальними знаннями",
            "missing": "вигаданий_підрядок_якого_немає_в_жодній_статті",
        },
        "Article 2": {
            "existing": "Профілювання варіанту із інвертованим списком",
            "missing": "вигаданий_підрядок_якого_немає_в_жодній_статті",
        },
    }
    # Prepare a dictionary for all measured execution times.
    results: dict[tuple[str, str], dict[str, float]] = {}

    print(
        f"Reading articles from {readme_path} and searching selected substrings."
    )
    print(
        f"Article 1: existing='{patterns['Article 1']['existing']}', "
        f"missing='{patterns['Article 1']['missing']}'"
    )
    print(
        f"Article 2: existing='{patterns['Article 2']['existing']}', "
        f"missing='{patterns['Article 2']['missing']}'"
    )
    # Check that the selected patterns have the intended presence in each article.
    for article_name, text in articles.items():
        # Process both selected patterns for the current article.
        for substring_type, pattern in patterns[article_name].items():
            # Use Python's built-in search as the expected correct position.
            expected_position = text.find(pattern)
            if (substring_type == "existing") != (expected_position >= 0):
                # Report an incorrectly selected test pattern.
                raise ValueError(
                    f"The {substring_type} pattern for {article_name} "
                    "does not match the article."
                )
            for algorithm in algorithms.values():
                # Verify every custom algorithm before measuring its speed.
                if algorithm(text, pattern) != expected_position:
                    raise RuntimeError(
                        f"{algorithm.__name__} returned an incorrect result "
                        f"for {article_name} ({substring_type})."
                    )

    # Print a table header before displaying measurements.
    print(
        f"{'Article':<12} {'Substring':<10} "
        + " ".join(f"{name:>22}" for name in algorithms)
    )

    # Measure each algorithm for every article and every pattern.
    for article_name, text in articles.items():
        # Process the existing and missing patterns separately.
        for substring_type, pattern in patterns[article_name].items():
            # Measure all algorithms on the same text and pattern.
            timings = {
                name: benchmark(algorithm, text, pattern)
                for name, algorithm in algorithms.items()
            }
            # Save the measurements for later average calculations.
            results[(article_name, substring_type)] = timings

            # Display each average time in microseconds for easier comparison.
            formatted_timings = [
                f"{timings[name] * 1_000_000:>19.3f} us"
                for name in algorithms
            ]
            print(
                f"{article_name:<12} {substring_type:<10} "
                + " ".join(formatted_timings)
            )
            # Select the algorithm with the smallest measured time.
            fastest_algorithm = min(timings, key=timings.get)
            print(f"  Fastest: {fastest_algorithm}")

    # Compare each article using the average of its existing and missing searches.
    print("\nAverage by article:")
    # Prepare average timings grouped by article.
    article_averages: dict[str, dict[str, float]] = {}
    for article_name in articles:
        # Average the existing and missing pattern measurements.
        article_averages[article_name] = {
            algorithm_name: sum(
                results[(article_name, substring_type)][algorithm_name]
                for substring_type in ("existing", "missing")
            )
            / 2
            for algorithm_name in algorithms
        }
        # Select the fastest algorithm for the current article.
        fastest_algorithm = min(
            article_averages[article_name].items(),
            key=lambda result: result[1],
        )[0]
        print(f"{article_name}: {fastest_algorithm}")

    # Average all four cases to determine the overall fastest implementation.
    # Calculate one average value for every algorithm.
    overall_averages = {
        algorithm_name: sum(
            results[case][algorithm_name] for case in results
        )
        / len(results)
        for algorithm_name in algorithms
    }
    # Select the fastest algorithm across all tests.
    fastest_overall = min(overall_averages, key=overall_averages.get)
    print(f"Overall: {fastest_overall}")


if __name__ == "__main__":
    main()
