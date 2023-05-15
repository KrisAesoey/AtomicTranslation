from re import L
from filehandler import FileHandler
from atomic_preprocessor import AtomicPreprocessor
from atomic_logifier import AtomicLogifier

import nltk


class AtomicGenerator():

    def __init__(self,
                 in_dir='./atomic_data/',
                 out_dir='./atomic_datasets/') -> None:
        self.filehandler = FileHandler(in_dir=in_dir, out_dir=out_dir)
        self.preprocessor = AtomicPreprocessor()
        self.logifier = AtomicLogifier()

    def untag_if_then(self, if_then_list: list[str]) -> list[str]:
        untagged = []

        for sentence in if_then_list:
            sentence = sentence.lower().replace(',', " ")
            tagged_tupes = [nltk.tag.str2tuple(t) for t in sentence.split()]
            untagged_sentence = [word for word, _ in tagged_tupes]
            untagged.append(" ".join(untagged_sentence))
        return untagged

    def generate_atomic_datasets(self, file_name: str):
        self.logifier.add_quantifiers = True
        data = self.filehandler.read_from_csv(file_name)
        if_then_all = self.preprocessor.preprocess_atomic(data[1:])
        if_then_categories = self.preprocessor.split__atomic_by_categories(
            if_then_all)

        print("---Creating dataset for all if-then relations---")
        logic = self.logifier.atomic_data_to_logic(if_then_all)
        self.filehandler.write_dataset_to_csv(
            self.untag_if_then(if_then_all), logic, "all_dataset")

        for category in if_then_categories.keys():
            print("---Creating dataset for " +
                  category + " if-then relations---")
            logic = self.logifier.atomic_data_to_logic(
                if_then_categories[category])
            self.filehandler.write_dataset_to_csv(
                self.untag_if_then(if_then_categories[category]), logic, category + "_dataset")

    def generate_atomic_datasets_wo_quantifiers(self, file_name: str):
        self.logifier.add_quantifiers = False
        data = self.filehandler.read_from_csv(file_name)
        if_then_all = self.preprocessor.preprocess_atomic(data[1:])
        if_then_categories = self.preprocessor.split__atomic_by_categories(
            if_then_all)

        print("---Creating dataset for all if-then relations---")
        logic = self.logifier.atomic_data_to_logic(if_then_all)
        self.filehandler.write_dataset_to_csv(
            self.untag_if_then(if_then_all), logic, "all_dataset_wo_q")

        for category in if_then_categories.keys():
            print("---Creating dataset for " +
                  category + " if-then relations---")
            logic = self.logifier.atomic_data_to_logic(
                if_then_categories[category])
            self.filehandler.write_dataset_to_csv(
                self.untag_if_then(if_then_categories[category]), logic, category + "_dataset_wo_q")


if __name__ == "__main__":
    ag = AtomicGenerator()
    ag.generate_atomic_datasets("v4_atomic_all.csv")
    ag.generate_atomic_datasets_wo_quantifiers("v4_atomic_all.csv")
