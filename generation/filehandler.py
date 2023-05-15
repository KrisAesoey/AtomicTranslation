import os
import csv


class FileHandler:
    def __init__(self, in_dir='./in/', out_dir='./out/') -> None:
        self.in_dir = in_dir
        self.out_dir = out_dir

    def read_from_csv(self, file_path: str) -> list[str]:
        """
        Reads csv file from input directory and returns
        a list with each line in file as entries.
        """
        data = []
        with open(self.in_dir + file_path) as file:
            reader = csv.reader(file, delimiter="\t")
            for line in reader:
                data.append(line)
        return data

    def write_list_to_csv(self, data: list[str], name: str) -> None:
        data_path = self.out_dir + name + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w', newline='') as data_file:
            writer = csv.writer(data_file, delimiter='\n')
            writer.writerow(data)

    def write_dataset_to_csv(self, contexts: list[str], targets: list[str], name: str) -> None:
        """
        Writes dataset into csv file where contexts
        and targets are seperated by tabs.
        """
        data_path = self.out_dir + name + '.csv'
        os.makedirs(os.path.dirname(data_path), exist_ok="True", mode=0o755)
        with open(data_path, 'w', newline='') as data_file:
            writer = csv.writer(data_file, delimiter='\t')
            for c, t in zip(contexts, targets):
                writer.writerow([c, t])
