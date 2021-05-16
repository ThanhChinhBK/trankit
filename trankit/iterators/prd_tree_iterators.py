import json
import os

from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co

from trankit.iterators.ner_iterators import Train_Instance
from trankit.utils.prd_tree_utils import load_prd_trees


class PrdTreeDataset(Dataset):
    """Dataset interface from """
    def __getitem__(self, index) -> T_co:
        return self.numberized_data[index]

    def __init__(self, config, prd_fpath, evaluate=False):
        self.config = config
        self.evaluate = evaluate
        self.numberized_data = []

        # load data
        self.config.vocab_fpath = os.path.join(self.config._save_dir, '{}.ner-vocab.json'.format(self.config.lang))
        self.data = load_prd_trees(self.config, prd_fpath, evaluate)

        if os.path.exists(self.config.vocab_fpath):
            with open(self.config.vocab_fpath) as f:
                self.vocabs = json.load(f)
        else:
            self.vocabs = {"UNK": 0}

    def __len__(self):
        return len(self.data)

    def numberize(self):
        numberized_data = []
        skip = 0
        for sentence in self.data:
            pieces_list = [[p for p in self.config.wordpiece_splitter.tokenize(w[0]) if p != '▁'] for w in sentence]
            for pieces in pieces_list:
                if len(pieces) == 0:
                    pieces += ['-']
            word_lens = [len(x) for x in pieces_list]
            assert 0 not in word_lens
            flat_pieces = [p for pieces in pieces_list for p in pieces]
            assert len(flat_pieces) > 0

            if len(flat_pieces) > self.config.max_input_length - 2:
                skip += 1
                continue

            # Pad word pieces with special tokens
            piece_idxs = self.config.wordpiece_splitter.encode(
                flat_pieces,
                add_special_tokens=True,
                max_length=self.config.max_input_length,
                truncation=True
            )

            attn_masks = [1] * len(piece_idxs)
            piece_idxs = piece_idxs
            assert len(piece_idxs) > 0

            entity_label_idxs = [self.vocabs[label] for label in sentence['entity-labels']]

            instance = Train_Instance(
                words=sentence['words'],
                word_num=len(sentence['words']),
                piece_idxs=piece_idxs,
                attention_masks=attn_masks,
                word_lens=word_lens,
                entity_label_idxs=entity_label_idxs
            )
            numberized_data.append(instance)
        print('Skipped {} over-length examples'.format(skip))
        print('Loaded {} examples'.format(len(numberized_data)))
        self.numberized_data = numberized_data
