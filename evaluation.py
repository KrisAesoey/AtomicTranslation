from tqdm import tqdm
import os
import csv
import numpy as np


def write_results_to_csv(predicted: list[str], golden: list[str], name: str) -> None:
    data_path = "./results/" + name + '.csv'
    os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
    with open(data_path, 'w', newline='') as data_file:
        writer = csv.writer(data_file, delimiter='\t')
        for c, t in zip(predicted, golden):
            writer.writerow([c, t])


def levenshteinDistance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1-1] == token2[t2-1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    return distances[len(token1)][len(token2)]


def average_ld(predicted_list: list[list[str]], golden_list: list[list[str]]) -> float:
    total = 0
    for predicted, golden in tqdm(zip(predicted_list, golden_list)):
        total += levenshteinDistance(predicted, golden)
    average = total / len(golden_list)
    return str(average)


def per_formula_accuracy(predicted: list[str], golden: list[str]) -> int:
    if predicted == golden:
        return 1
    return 0


def average_formula_accuracy(predicted_list: list[list[str]], golden_list: list[list[str]], write_results=False) -> float:
    total = 0
    correct_predictions = []
    correct_golden = []
    wrong_predictions = []
    wrong_golden = []
    for predicted, golden in zip(predicted_list, golden_list):
        result = per_formula_accuracy(predicted, golden)
        if result:
            correct_predictions.append(predicted)
            correct_golden.append(golden)
        else:
            wrong_predictions.append(predicted)
            wrong_golden.append(golden)
        total += result
    average = total / len(golden_list)
    if write_results:
        write_results_to_csv(correct_predictions,
                             correct_golden, "correct_formulas")
        write_results_to_csv(wrong_predictions, wrong_golden, "wrong_formulas")
    return str(average)


def per_token_accuracy(predicted: list[str], golden: list[str]) -> float:
    correct = 0
    for i in range(len(golden)):
        if i < len(predicted):
            if golden[i] == predicted[i]:
                correct += 1
    return correct / len(golden)


def average_token_accuracy(predicted_list: list[list[str]], golden_list: list[list[str]]) -> float:
    total = 0
    for predicted, golden in zip(predicted_list, golden_list):
        total += per_token_accuracy(predicted, golden)
    average = total / len(golden_list)
    return str(average)


def test_lev():
    t1 = (["hi"], ["ho"])
    t2 = ([], ["ay", "yo"])
    t3 = (["ay", "yo", "ma"], [])
    t4 = (["a", "b", "c"], ["a", "c"])
    t5 = (["a", "b", "c", "d"], ["a", "b", "c", "d"])
    assert levenshteinDistance(t1[0], t1[1]) == 1
    assert levenshteinDistance(t2[0], t2[1]) == 2
    assert levenshteinDistance(t3[0], t3[1]) == 3
    assert levenshteinDistance(t4[0], t4[1]) == 1
    assert levenshteinDistance(t5[0], t5[1]) == 0
    print("levenshtein passed")


def test_formula():
    t1 = ([["hi"]], [["ho"]])
    t2 = ([["hi"]], [["hi"]])
    t3 = ([["ay"]], [[]])
    assert average_formula_accuracy(t1[0], t1[1]) == 0
    assert average_formula_accuracy(t2[0], t2[1]) == 1
    assert average_formula_accuracy(t3[0], t3[1]) == 0
    print("formula passed")


def test_token():
    t1 = ([["hi"]], [["ho"]])
    t2 = ([["hei"], ["da"]], [["hei"], ["nyet"]])
    assert (average_token_accuracy(t1[0], t1[1])) == 0
    assert average_token_accuracy(t2[0], t2[1]) == 0.5
    print("token passed")


if __name__ == "__main__":
    test_lev()
    test_formula()
    test_token()
