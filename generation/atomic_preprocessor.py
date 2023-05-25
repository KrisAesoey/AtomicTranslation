import enum
import re
import nltk
from typing import Tuple
from tqdm import tqdm

from filehandler import FileHandler


class POStagger:
    """
    Class meant for taking if-then relations from Atomic
    and part-of-speech tagging them.
    """

    def pos_tag_sentence(self, sentence: str) -> str:
        sentence = sentence.split()
        sentence_tag_tuples = nltk.pos_tag(sentence)
        tagged_sentence = []
        for word, tag in sentence_tag_tuples:
            if word.lower() in ["personx", "persony", "personz", "personx's", "persony's", "personz's"]:
                tagged_sentence.append(word + "/" + "IND")
            else:
                tagged_sentence.append(word + "/" + tag)
        return " ".join(tagged_sentence)

    def pos_tag_if_then_relation(self, if_then: str) -> str:
        event, relation, inference = if_then.split(',')
        event_tagged = self.pos_tag_sentence(event)
        inference_tagged = self.pos_tag_sentence(inference)
        return ",".join([event_tagged, relation, inference_tagged])

    def pos_tag_if_then_relations(self, if_then_list: list[str]) -> list[str]:
        tagged_if_then_relations = []
        print("---POS-tagging if-then relations---")
        for if_then in tqdm(if_then_list):
            tagged_if_then = self.pos_tag_if_then_relation(if_then)
            tagged_if_then_relations.append(tagged_if_then)
        return tagged_if_then_relations


class AtomicPreprocessor:

    def __init__(self,
                 in_dir='./atomic_data/', out_dir='./generated/') -> None:
        self.filehandler = FileHandler(in_dir=in_dir, out_dir=out_dir)
        self.POStagger = POStagger()
        self.relations = [
            "oEffect",
            "oReact",
            "oWant",
            "xAttr",
            "xEffect",
            "xIntent",
            "xNeed",
            "xReact",
            "xWant",
        ]
        self.categories = {
            "persona": ["xAttr"],
            "mental": ["xIntent", "xReact", "oReact"],
            "event": ["xEffect", "oEffect", "xNeed", "xWant", "oWant"]
        }

    def split_open_closed(self, data: list[str], return_open=False) -> None:
        """
        Splits a given dataset into a closed and open dataset,
        based on if there exists an unknown word in it or not.
        Returns the closed data, unless specified to return the open.
        """
        open_data = []
        closed_data = []

        if return_open:
            print("---Obtaining open if-then-inferences----")
        else:
            print("---Obtaining closed if-then-inferences---")

        for d in tqdm(data):
            sentence = d[0]
            if "___" in sentence:
                open_data.append(sentence)
            else:
                closed_data.append(sentence)

        if return_open:
            return open_data

        return closed_data

    def correct_individuals(self, inference):
        """
        Standardizes and corrects the spellings of the
        individuals PersonX and PersonY in the Atomic dataset.
        The spell correction handles all cases of an edit distance
        of 1 from the correct spelling.
        """

        # all versions of the words person of Edit Distance 1
        person_spellings = ["person", "eprson", "preson", "pesron", "perosn", "persno", "perons",
                            "erson", "prson", "peson", "peron", "perso"]

        words = inference.lower().replace('.', "").split()
        corrected_inference = []
        i = 0
        while i < len(words):
            # check if person has been been misspelled and a space
            # exists between it and the individual variable
            if words[i] in person_spellings:
                if i != len(words) - 1:
                    if words[i+1] == 'x':
                        corrected_inference.append("PersonX")
                        i += 2
                        continue
                    elif words[i+1] == 'y':
                        corrected_inference.append("PersonY")
                        i += 2
                        continue
                    elif words[i+1] == 'z':
                        corrected_inference.append("PersonZ")
                        i += 2
                        continue
                    elif words[i+1] == "x's" or words[i+1] == "xs":
                        corrected_inference.append("PersonX's")
                        i += 2
                        continue
                    elif words[i+1] == "y's" or words[i+1] == "ys":
                        corrected_inference.append("PersonY's")
                        i += 2
                        continue
                    elif words[i+1] == "z's" or words[i+1] == "zs":
                        corrected_inference.append("PersonZ's")
                        i += 2
                        continue
                    else:
                        corrected_inference.append("person")
                else:
                    corrected_inference.append("person")

            else:
                # check for misspelling of personx, persony and personz
                for person_spelling in person_spellings:
                    if words[i] == person_spelling + "x":
                        corrected_inference.append("PersonX")
                        break
                    elif words[i] == person_spelling + "y":
                        corrected_inference.append("PersonY")
                        break

                    elif words[i] == person_spelling + "z":
                        corrected_inference.append("PersonZ")
                        break

                    elif words[i] == person_spelling + "xs" or words[i] == person_spelling + "x's":
                        corrected_inference.append("PersonX's")
                        break
                    elif words[i] == person_spelling + "ys" or words[i] == person_spelling + "y's":
                        corrected_inference.append("PersonY's")
                        break

                    elif words[i] == person_spelling + "zs" or words[i] == person_spelling + "z's":
                        corrected_inference.append("PersonZ's")
                        break

                else:
                    # check if the "person" part has been omitted
                    if words[i] == 'x':
                        corrected_inference.append("PersonX")
                    elif words[i] == 'y':
                        corrected_inference.append("PersonY")
                    elif words[i] == 'z':
                        corrected_inference.append("PersonZ")

                    elif words[i] == "x's" or words[i] == "xs":
                        corrected_inference.append("PersonX's")
                    elif words[i] == "y's" or words[i] == "ys":
                        corrected_inference.append("PersonY's")
                    elif words[i] == "z's" or words[i] == "zs":
                        corrected_inference.append("PersonZ's")
                    else:
                        corrected_inference.append(words[i])
            i += 1

        return " ".join(corrected_inference)

    def exclude_example(self, event, inference, remove_none=True):
        """
        Given an event and inference determine of they
        should be excluded, based on if PersonZ is involved,
        or the inference is empty.
        Inferences of None can also be selected to be removed.
        """
        if "personz" in event.lower() or "personz" in inference.lower():
            return True
        if inference == "":
            return True
        if remove_none and inference == "none":
            return True
        else:
            return False

    def split_inferences_into_list(self, inferences_prefix_set) -> list[list[str]]:
        """
        Splits the input format of the inferences + prefix + dataset_tag to
        just be a list of all nine relation types, where each entry is a list
        consisting of that relations inferences
        """
        ips = inferences_prefix_set.replace("\"", "")
        ips = re.split(',(?! .+\])', ips)
        return ips[0:9]

    def split_into_if_then(self, data: str, remove_none=True) -> list[str]:
        if_then_list = []
        event, *inferences_prefix_set = data.split(',', maxsplit=1)
        # remove unexpected " from events
        event = event.replace('"', "")
        inference_list = self.split_inferences_into_list(
            inferences_prefix_set[0])
        for index, inferences in enumerate(inference_list):
            if inferences != "[]":
                stripped_inferences = inferences.replace(
                    '[', "").replace(']', "")
                stripped_inferences = [x.strip()
                                       for x in stripped_inferences.split(',')]
                for inference in stripped_inferences:
                    corrected_inference = self.correct_individuals(inference)
                    # skip examples that are not desired
                    if self.exclude_example(event, corrected_inference, remove_none):
                        continue
                    if_then_list.append(
                        ",".join([event, self.relations[index], corrected_inference]))

        return if_then_list

    def preprocess_atomic(self, data: list[str], open_data=False, remove_none=True) -> list[str]:
        # split the Atomic data and only keep the closed set
        data = self.split_open_closed(data, return_open=open_data)

        # reformat from all inferences of the event being in one list
        # into every index being a seperate if-then relation
        if_then_relations = []
        print("---Splitting into if-then relations---")
        for if_then_collection in tqdm(data):
            if_then_list = self.split_into_if_then(
                if_then_collection, remove_none=remove_none)
            for if_then_relation in if_then_list:
                if_then_relations.append(if_then_relation)

        # POS tag the if-then relations
        tagged_if_then_relations = self.POStagger.pos_tag_if_then_relations(
            if_then_relations)
        return tagged_if_then_relations

    def split__atomic_by_categories(self, if_then_list: list[str]) -> dict[str]:
        if_then_categories = {k: [] for k in self.categories.keys()}

        for if_then in if_then_list:
            _, dim, _ = if_then.split(',')
            for key in self.categories.keys():
                if dim in self.categories[key]:
                    if_then_categories[key].append(if_then)
        return if_then_categories

    def read_data_write_dataset(self, filename: str, open_data=False, remove_none=True) -> None:
        # get the Atomic data, and only keep the closed set
        data = self.filehandler.read_from_csv(filename)
        data = data[1:]  # remove header
        data = self.preprocess_atomic(
            data, open_data=open_data, remove_none=remove_none)
        if open_data:
            self.filehandler.write_list_to_csv(data, "atomic_open")
        else:
            self.filehandler.write_list_to_csv(data, "atomic_closed")


if __name__ == "__main__":
    ap = AtomicPreprocessor()
    ap.read_data_write_dataset("v4_atomic_all.csv")
