import nltk
from sortedcollections import OrderedSet

from filehandler import FileHandler


class AtomicLogifier:
    """
    Class designed to create logical formulas from if-then relations
    found in the Atomic knowledge base. The input relations have to be in
    the format: "event,relation_dimension,inference" and the output formulas
    are atoms "predicate (variable)" conjuncted by "&", where the event is the
    body, and the inference is the head.
    """

    def __init__(self,
                 in_dir='./generated/',
                 out_dir='./atomic_dataset/',
                 add_quantifiers=True) -> None:
        self.add_quantifiers = add_quantifiers

    def event_to_logic(self, event: list[str]) -> tuple[list[str], list[str]]:
        event_logic = []
        tagged_tupes = [nltk.tag.str2tuple(t) for t in event.split()]

        individuals = OrderedSet([word[0:7].lower()
                                  for word, tag in tagged_tupes if tag == "IND"])

        # create logic of all individuals in the event and store variables
        variables = []
        for ind in individuals:
            variables.append(ind[6])
            event_logic.append("person " + "(" + ind[6] + ")")

        # find the other atoms of the event
        verb_atom = []
        object_atom = []
        verb_atom_found = False
        # remove the subject Person X as it has been added already
        tagged_tupes.pop(0)
        for word, tag in tagged_tupes:
            word = word.lower()
            if tag in ["IND", "DT"]:
                verb_atom_found = True
            # check if verb part has a word already due to faulty tagging by nltk
            elif verb_atom and tag in ["NN", "NNS", "JJ", "JJS"]:
                verb_atom_found = True
                object_atom.append(word)
            elif not verb_atom_found:
                verb_atom.append(word)
            else:
                object_atom.append(word)

        verb = " ".join(verb_atom)

        # check arity of the verb atom of the event
        # depending on amount of individuals available
        if len(individuals) == 2:
            event_logic.append(verb + " (x,z,y)")
        else:
            event_logic.append(verb + " (x,z)")

        # add the object atom if there is one
        if object_atom:
            obj = " ".join(object_atom)
            event_logic.append(obj + " (z)")
        variables.append("z")

        return event_logic, variables

    def inference_to_logic(self, inference: list[str], event_logic: list[str], subject="x") -> tuple[list[str], list[str]]:
        inference_logic = []
        tagged_tupes = [nltk.tag.str2tuple(t) for t in inference.split()]

        individuals = OrderedSet([word[0:7].lower()
                                  for word, tag in tagged_tupes if tag == "IND"])

        verb_atom = []
        verb_atom_done = False
        atoms = []
        current_atom = []
        concept_variables = []
        current_var = "a"

        # create logic of all individuals in the inference
        # and check if they are already mentioned in the event
        for ind in individuals:
            inference_individual = "person " + "(" + ind[6] + ")"
            if inference_individual not in event_logic:
                inference_logic.append(inference_individual)
                concept_variables.append(ind[6])

        for word, tag in tagged_tupes:
            word = word.lower()
            if tag in ["IND", "DT", "CC", "PRP", "PRP$"]:
                if verb_atom_done and current_atom:
                    # finish current concept and create new one
                    atoms.append(
                        " ".join(current_atom) + " (" + current_var + ")")
                    concept_variables.append(current_var)
                    current_var = chr(ord(current_var) + 1)
                    current_atom = []
                if verb_atom:
                    verb_atom_done = True

            elif not verb_atom_done:
                verb_atom.append(word)

            else:
                current_atom.append(word)
        # add last found concept to the list as well
        if current_atom:
            atoms.append(
                " ".join(current_atom) + " (" + current_var + ")")
            concept_variables.append(current_var)

        redundant_concept = False
        # add the verb part correctly
        if verb_atom:
            verb = " ".join(verb_atom)
            if subject == "x":
                # if no new atoms are added
                if not atoms:
                    # but both person x and y is in the inference
                    if len(individuals) == 2:
                        inference_logic.append(verb + " (x,y)")
                    else:
                        inference_logic.append(verb + " (x)")

                if atoms:
                    # check if the only new atom is referencing the
                    # object atom found in the event, and if so
                    # make the inference logic refer to it
                    if len(atoms) == 1 and atoms[0][:4] + " (z)" in event_logic:
                        redundant_concept = True
                        if len(individuals) == 2:
                            inference_logic.append(
                                verb + " (x,z,y)")
                        else:
                            inference_logic.append(verb + " (x,z)")
                        concept_variables = []
                    # but both person x and y is in the inference
                    elif len(individuals) == 2:
                        for cv in concept_variables:
                            inference_logic.append(
                                verb + " (x," + cv + ",y)")
                    else:
                        for cv in concept_variables:
                            inference_logic.append(
                                verb + " (x," + cv + ")")
            else:
                if "person (y)" in event_logic:
                    # if no new atoms are added
                    if not atoms:
                        # but both person x and y is in the inference
                        if len(individuals) == 2:
                            inference_logic.append(verb + " (y,x)")
                        else:
                            inference_logic.append(verb + " (y)")

                    # check if the only new atom is referencing the
                    # object atom found in the event, and if so
                    # make the inference logic refer to it
                    elif len(atoms) == 1 and atoms[0][:4] + " (z)" in event_logic:
                        redundant_concept = True
                        if len(individuals) == 2:
                            inference_logic.append(
                                verb + " (y,z,x)")
                        else:
                            inference_logic.append(verb + " (y,z)")
                    # but both person x and y is in the inference
                    elif len(individuals) == 2:
                        for cv in concept_variables:
                            inference_logic.append(
                                verb + " (y," + cv + ",x)")
                    else:
                        for cv in concept_variables:
                            inference_logic.append(
                                verb + " (y," + cv + ")")
                else:
                    # if no new atoms are added
                    if not atoms:
                        # but person x in inference
                        if len(individuals) == 1:
                            inference_logic.append(verb + " (u,x)")
                        else:
                            inference_logic.append(verb + " (u)")

                    # check if the only new atom is referencing the
                    # object atom found in the event, and if so
                    # make the inference logic refer to it
                    elif len(atoms) == 1 and atoms[0][:4] + " (z)" in event_logic:
                        redundant_concept = True
                        if len(individuals) == 2:
                            inference_logic.append(
                                verb + " (u,z,x)")
                        else:
                            inference_logic.append(verb + " (u,z)")
                        concept_variables = []
                    # but person x is in the inference
                    elif len(individuals) == 1:
                        for cv in concept_variables:
                            inference_logic.append(
                                verb + " (u," + cv + ",x)")
                    else:
                        for cv in concept_variables:
                            inference_logic.append(
                                verb + " (u," + cv + ")")
                    concept_variables.append("u")

        if not redundant_concept:
            for atom in atoms:
                if atom not in event_logic:
                    inference_logic.append(atom)

        return inference_logic, concept_variables

    def atomic_if_then_to_logic(self, if_then: str) -> str:
        """
        Given a if-then-relation in the format
        "event,relation_dimension,inference" returns
        an equivalent logical formula.
        """
        event, dim, inference = if_then.split(',')
        event_logic, event_vars = self.event_to_logic(event)

        # construct the body by conjuncting the atoms in the event
        body_logic = " & ".join(event_logic)

        inference_logic, inference_vars = self.inference_to_logic(
            inference, event_logic, subject=dim[0])

        # construct the body by conjuncting the atoms in the event
        head_logic = " & ".join(inference_logic)

        if self.add_quantifiers:
            universal_vars = " ".join(event_vars)
            if inference_vars:
                exist_vars = " ".join(inference_vars)
                head_logic = "E " + exist_vars + " ( " + head_logic + " )"
            if_then_logic = "A " + universal_vars + \
                " ( ( " + body_logic + " ) -> " + head_logic + " )"

        else:
            if_then_logic = body_logic + " -> " + head_logic

        return if_then_logic

    def atomic_data_to_logic(self, data: list[str]) -> list[str]:
        """
        Given a list of  if-then relations in the format "event,relation_dimension,inference"
        returns their equivalent logical formulas in a list.
        """
        logic_data = []
        for d in data:
            l = self.atomic_if_then_to_logic(d)
            logic_data.append(l)

        return logic_data

    def read_data_write_dataset(self, dataset_name: str) -> None:
        """
        Loads Atomic data from input directory
        and creates dataset with logical formulas in
        output directory.
        """
        text_data = self.filehandler.read_from_csv(
            dataset_name + ".csv")
        logic_data = self.atomic_data_to_logic(text_data)

        self.filehandler.write_dataset_to_csv(
            text_data, logic_data, dataset_name + "_logic")
