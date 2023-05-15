from filehandler import FileHandler
from typing import Tuple
import nltk
from tqdm import tqdm


class DketGenerator():

    def __init__(self, in_dir='./dket_data/', out_dir='./dket_datasets/') -> None:
        self.filehandler = FileHandler(in_dir=in_dir, out_dir=out_dir)

    def _logic_replace_indices(self, text: list[str], logic: list[str]) -> list[str]:
        """Replaces logic where words are represented as indices
        in original sentences with the actual word.  
        """
        corrected_logic = []
        for token in logic:
            if token.isdigit():
                corrected_logic.append(text[int(token)])
            else:
                corrected_logic.append(token)
        return corrected_logic

    def _remove_pos_tags(self, text: str) -> list[str]:
        """
        Given text in format "word1/POS word2/POS"
        removes <EOS> token and removes POS.
        """
        text = text.replace("<EOS>/<EOS>", "")
        text_tups = [nltk.tag.str2tuple(t) for t in text.split()]
        text = [t[0] for t in text_tups]
        return text

    def clean_data(self, text: str, logic: str) -> Tuple[str, str]:
        """Transforms text in word/POS format and
        logic in index reference format to clean text
        and logic for use in Transformers."""
        text = self._remove_pos_tags(text)

        # Remove <EOS> tag and the LOC# identifier for indices
        logic = logic.replace("LOC#", "").replace("<EOS>", "")
        logic_list = logic.split()

        corrected_logic = self._logic_replace_indices(text, logic_list)
        return (" ".join(text), " ".join(corrected_logic))

    def read_data_write_dataset(self, dataset_name: str) -> None:
        """
        Loads DKET data from input directory
        and creates dataset with clean format ii 
        output directory.
        """
        training_data = self.filehandler.read_from_csv(
            dataset_name + ".tsv")
        text_data = []
        logic_data = []
        for text, logic in tqdm(training_data):
            td, ld = self.clean_data(text, logic)
            text_data.append(td)
            logic_data.append(ld)

        self.filehandler.write_dataset_to_csv(
            text_data, logic_data, "dket_" + dataset_name)


if __name__ == "__main__":
    dg = DketGenerator()
    datasets = ["2k", "5k", "10k", "20k"]  # all dataset sizes
    for d in datasets:
        dg.read_data_write_dataset("train_" + d)
        dg.read_data_write_dataset("validation_" + d)
