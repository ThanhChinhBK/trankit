import dataclasses
import json
from typing import List, Optional, Tuple

import nltk
import torch
from nltk.corpus import BracketParseCorpusReader
import spacy_alignments as tokenizations


PTB_UNESCAPE_MAPPING = {
    "«": '"',
    "»": '"',
    "‘": "'",
    "’": "'",
    "“": '"',
    "”": '"',
    "„": '"',
    "‹": "'",
    "›": "'",
    "\u2013": "--",  # en dash
    "\u2014": "--",  # em dash
}
NO_SPACE_BEFORE = {"-RRB-", "-RCB-", "-RSB-", "''"} | set("%.,!?:;")
NO_SPACE_AFTER = {"-LRB-", "-LCB-", "-LSB-", "``", "`"} | set("$#")
NO_SPACE_BEFORE_TOKENS_ENGLISH = {"'", "'s", "'ll", "'re", "'d", "'m", "'ve"}
PTB_DASH_ESCAPED = {"-RRB-", "-RCB-", "-RSB-", "-LRB-", "-LCB-", "-LSB-", "--"}


class Treebank(torch.utils.data.Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    @property
    def trees(self):
        return [x.tree for x in self.examples]

    @property
    def sents(self):
        return [x.words for x in self.examples]

    @property
    def tagged_sents(self):
        return [x.pos() for x in self.examples]

    def filter_by_length(self, max_len):
        return Treebank([x for x in self.examples if len(x.leaves()) <= max_len])

    def without_gold_annotations(self):
        return Treebank([x.without_gold_annotations() for x in self.examples])


def read_text(text_path):
    sents = []
    sent = []
    end_of_multiword = 0
    multiword_combined = ""
    multiword_separate = []
    multiword_sp_after = False
    with open(text_path) as f:
        for line in f:
            if not line.strip() or line.startswith("#"):
                if sent:
                    sents.append(([w for w, sp in sent], [sp for w, sp in sent]))
                    sent = []
                    assert end_of_multiword == 0
                continue
            fields = line.split("\t", 2)
            num_or_range = fields[0]
            w = fields[1]

            if "-" in num_or_range:
                end_of_multiword = int(num_or_range.split("-")[1])
                multiword_combined = w
                multiword_separate = []
                multiword_sp_after = "SpaceAfter=No" not in fields[-1]
                continue
            elif int(num_or_range) <= end_of_multiword:
                multiword_separate.append(w)
                if int(num_or_range) == end_of_multiword:
                    _, separate_to_combined = tokenizations.get_alignments(
                        multiword_combined, multiword_separate
                    )
                    have_up_to = 0
                    for i, char_idxs in enumerate(separate_to_combined):
                        if i == len(multiword_separate) - 1:
                            word = multiword_combined[have_up_to:]
                            sent.append((word, multiword_sp_after))
                        elif char_idxs:
                            word = multiword_combined[have_up_to: max(char_idxs) + 1]
                            sent.append((word, False))
                            have_up_to = max(char_idxs) + 1
                        else:
                            sent.append(("", False))
                    assert int(num_or_range) == len(sent)
                    end_of_multiword = 0
                    multiword_combined = ""
                    multiword_separate = []
                    multiword_sp_after = False
                continue
            else:
                assert int(num_or_range) == len(sent) + 1
                sp = "SpaceAfter=No" not in fields[-1]
                sent.append((w, sp))
    return sents


def load_prd_trees(config, ptr_path, evaluate):
    """Load a treebank from ptr file format.

    This function update tag vocab when evaluate is True.
    """
    reader = BracketParseCorpusReader("", [ptr_path])
    trees = reader.parsed_sents()
    sentences = []
    for tree in trees:
        words = ptb_unescape(tree.leaves())
        print(words)
        sp_after = guess_space_after(tree.leaves())
        sentences.append((words, sp_after))

    assert len(trees) == len(sentences)
    treebank = Treebank(
        [
            ParsingExample(tree=tree, words=words, space_after=space_after)
            for tree, (words, space_after) in zip(trees, sentences)
        ]
    )
    loaded_sentences = []
    for example in treebank:
        words = example.words
        labels = [token[1] for token in example.tree.pos()]
        loaded_sentences.append({"words": words, "ptr_labels": labels})
        assert len(example.words) == len(example.leaves()), (
            "Constituency tree has a different number of tokens than the CONLL-U or "
            "other file used to specify reversible tokenization."
        )
    if not evaluate:
        tag_set = set()
        for sentence in loaded_sentences:
            tag_set.update(sentence["ptr_labels"])
        tag_list = list(tag_set)
        vocab = {"UNK": 0}
        tag_list = [t for t in tag_list if t != 'O']
        tag_list.sort()
        for t in tag_list:
            vocab[t] = vocab.get(t, len(vocab))

        with open(config.vocab_fpath,
                  'w') as f:
            json.dump(vocab, f)
    return loaded_sentences


def guess_space_after(escaped_words, for_english=True):
    if not for_english:
        return guess_space_after_non_english(escaped_words)

    sp_after = [True for _ in escaped_words]
    for i, word in enumerate(escaped_words):
        if word.lower() == "n't" and i > 0:
            sp_after[i - 1] = False
        elif word.lower() == "not" and i > 0 and escaped_words[i - 1].lower() == "can":
            sp_after[i - 1] = False

        if i > 0 and (
            (
                word.startswith("-")
                and not any(word.startswith(x) for x in PTB_DASH_ESCAPED)
            )
            or any(word.startswith(x) for x in NO_SPACE_BEFORE)
            or word.lower() in NO_SPACE_BEFORE_TOKENS_ENGLISH
        ):
            sp_after[i - 1] = False
        if (
            word.endswith("-") and not any(word.endswith(x) for x in PTB_DASH_ESCAPED)
        ) or any(word.endswith(x) for x in NO_SPACE_AFTER):
            sp_after[i] = False

    return sp_after


def guess_space_after_non_english(escaped_words):
    sp_after = [True for _ in escaped_words]
    for i, word in enumerate(escaped_words):
        if i > 0 and (
                (
                        word.startswith("-")
                        and not any(word.startswith(x) for x in PTB_DASH_ESCAPED)
                )
                or any(word.startswith(x) for x in NO_SPACE_BEFORE)
                or word == "'"
        ):
            sp_after[i - 1] = False
        if (
                word.endswith("-") and not any(word.endswith(x) for x in PTB_DASH_ESCAPED)
        ) or any(word.endswith(x) for x in NO_SPACE_AFTER):
            sp_after[i] = False

    return sp_after


def ptb_unescape(words):
    cleaned_words = []
    for word in words:
        word = PTB_UNESCAPE_MAPPING.get(word, word)
        # This un-escaping for / and * was not yet added for the
        # parser version in https://arxiv.org/abs/1812.11760v1
        # and related model releases (e.g. benepar_en2)
        word = word.replace("\\/", "/").replace("\\*", "*")
        # Mid-token punctuation occurs in biomedical text
        word = word.replace("-LSB-", "[").replace("-RSB-", "]")
        word = word.replace("-LRB-", "(").replace("-RRB-", ")")
        word = word.replace("-LCB-", "{").replace("-RCB-", "}")
        word = word.replace("``", '"').replace("`", "'").replace("''", '"')
        cleaned_words.append(word)
    return cleaned_words


@dataclasses.dataclass
class ParsingExample:
    """A single parse tree and sentence."""

    words: List[str]
    space_after: List[bool]
    tree: Optional[nltk.Tree] = None
    _pos: Optional[List[Tuple[str, str]]] = None

    def leaves(self):
        if self.tree is not None:
            return self.tree.leaves()
        elif self._pos is not None:
            return [word for word, tag in self._pos]
        else:
            return None

    def pos(self):
        if self.tree is not None:
            return self.tree.pos()
        else:
            return self._pos

    def without_gold_annotations(self):
        return dataclasses.replace(self, tree=None, _pos=self.pos())
