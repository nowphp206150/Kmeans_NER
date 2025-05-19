from collections import Counter
from prettytable import PrettyTable


def print_table(data):

    table = PrettyTable()
    table.field_names = ['Entity', 'Precision', 'Recall', 'F1']

    for entity, metrics in data.items():
        table.add_row([entity, metrics['precision'],
                      metrics['recall'], metrics['f1']])

    print(table)


class SpanEntityScore(object):
    def __init__(self, id2label, is_nest=False, markup='bio'):
        self.id2label = id2label
        self.is_nest = is_nest
        self.markup = markup
        self.reset()

    def reset(self):
        self.origins = []
        self.founds = []
        self.rights = []

    def compute(self, origin, found, right):
        recall = 0 if origin == 0 else (right / origin)
        precision = 0 if found == 0 else (right / found)
        f1 = 0. if recall + \
            precision == 0 else (2 * precision * recall) / (precision + recall)
        return recall, precision, f1

    def result(self):
        class_info = {}
        origin_counter = Counter([x[0] for x in self.origins])
        found_counter = Counter([x[0] for x in self.founds])
        right_counter = Counter([x[0] for x in self.rights])
        for type_, count in origin_counter.items():
            origin = count
            found = found_counter.get(type_, 0)
            right = right_counter.get(type_, 0)
            recall, precision, f1 = self.compute(origin, found, right)
            class_info[type_] = {"precision": round(
                precision, 4), 'recall': round(recall, 4), 'f1': round(f1, 4)}
        origin = len(self.origins)
        found = len(self.founds)
        right = len(self.rights)
        recall, precision, f1 = self.compute(origin, found, right)
        return {'precision': precision, 'recall': recall, 'f1': f1}, class_info

    def remove_partial_overlap(self, pre_entities):
        pre_entities = [sorted(p, key=lambda x: x[3], reverse=True)
                        for p in pre_entities]

        def is_overlap(e1, entities):
            for e2 in entities:
                if (e1[1] < e2[1] and e2[1] <= e1[2] and e2[2] > e1[2]) or (e2[1] < e1[1] and e1[1] <= e2[2] and e1[2] > e2[2]):
                    return True
            return False
        not_overlap_entities = []
        for pre in pre_entities:
            not_overlap_entity = []
            for p in pre:
                if not is_overlap(p, not_overlap_entity):
                    not_overlap_entity.append(p[:-1])
            not_overlap_entities.append(not_overlap_entity)
        return not_overlap_entities

    def remove_overlap(self, pre_entities):
        pre_entities = [sorted(p, key=lambda x: x[3], reverse=True)
                        for p in pre_entities]

        def is_overlap(e1, entities):
            for e2 in entities:
                if e1[2] < e2[1] or e2[2] < e1[1]:
                    continue
                else:
                    return True
            return False
        not_overlap_entities = []
        for pre in pre_entities:
            not_overlap_entity = []
            for p in pre:
                if not is_overlap(p, not_overlap_entity):
                    not_overlap_entity.append(p[:-1])
            not_overlap_entities.append(not_overlap_entity)
        return not_overlap_entities

    def eval(self, label_entities, pre_entities):

        if self.is_nest:
            # nest eval
            pre_entities = self.remove_partial_overlap(pre_entities)
        else:
            # flat eval
            pre_entities = self.remove_overlap(pre_entities)
        for label, pre in zip(label_entities, pre_entities):
            self.origins.extend(label)
            self.founds.extend(pre)
            self.rights.extend(
                [p for p in pre if p in label])
        # print(self.origins, self.founds, self.rights)

    def update(self, label_paths, pred_paths):
        '''
        labels_paths: [[],[],[],....]
        pred_paths: [[],[],[],.....]

        :param label_paths:
        :param pred_paths:
        :return:
        Example:
            >>> labels_paths = [['O', 'O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
            >>> pred_paths = [['O', 'O', 'B-MISC', 'I-MISC', 'I-MISC', 'I-MISC', 'O'], ['B-PER', 'I-PER', 'O']]
        '''
        for label_path, pre_path in zip(label_paths, pred_paths):
            label_entities = self.get_entities(label_path, self.markup)
            pre_entities = self.get_entities(pre_path, self.markup)
            self.origins.extend(label_entities)
            self.founds.extend(pre_entities)
            self.rights.extend(
                [pre_entity for pre_entity in pre_entities if pre_entity in label_entities])

    def get_entities(self, seq, markup='bio'):
        '''
        equal to seqeval
        :param seq:
        :param id2label:
        :param markup:
        :return:
        '''
        assert markup in ['io', 'bio']
        if markup == 'bio':
            return self.get_entity_bio(seq)
        else:
            return self.get_entity_io(seq)

    def get_entity_bio(self, seq):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (span_type, span_start, span_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_bio(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        spans = []
        span = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = self.id2label[tag]
            if tag.startswith("B-"):
                if span[2] != -1:
                    spans.append(span)
                span = [-1, -1, -1]
                span[1] = indx
                # span[0] = tag.split('-')[1]
                span[0] = '-'.join(tag.split('-')[1:])
                span[2] = indx
                if indx == len(seq) - 1:
                    spans.append(span)
            elif tag.startswith('I-') and span[1] != -1:
                # _type = tag.split('-')[1]
                _type = '-'.join(tag.split('-')[1:])
                if _type == span[0]:
                    span[2] = indx

                if indx == len(seq) - 1:
                    spans.append(span)
            else:
                if span[2] != -1:
                    spans.append(span)
                span = [-1, -1, -1]
        return spans

    def get_entity_io(self, seq):
        """Gets entities from sequence.
        note: BIO
        Args:
            seq (list): sequence of labels.
        Returns:
            list: list of (span_type, span_start, span_end).
        Example:
            seq = ['B-PER', 'I-PER', 'O', 'B-LOC']
            get_entity_io(seq)
            #output
            [['PER', 0,1], ['LOC', 3, 3]]
        """
        def switch_to_BIO(labels):
            '''
            same with C2FNER
            '''
            past_label = 'O'
            labels_BIO = []
            for label in labels:
                if label.startswith('I-') and (past_label == 'O' or past_label[2:] != label[2:]):
                    labels_BIO.append('B-' + label[2:])
                else:
                    labels_BIO.append(label)
                past_label = label
            return labels_BIO
        if not isinstance(seq[0], str):
            seq = switch_to_BIO([self.id2label[s] for s in seq])
        spans = []
        span = [-1, -1, -1]
        for indx, tag in enumerate(seq):
            if not isinstance(tag, str):
                tag = self.id2label[tag]
            if tag.startswith("B-"):
                if span[2] != -1:
                    spans.append(span)
                span = [-1, -1, -1]
                span[1] = indx
                # span[0] = tag.split('-')[1]
                span[0] = '-'.join(tag.split('-')[1:])
                span[2] = indx
                if indx == len(seq) - 1:
                    spans.append(span)
            elif tag.startswith('I-') and span[1] != -1:
                # _type = tag.split('-')[1]
                _type = '-'.join(tag.split('-')[1:])
                if _type == span[0]:
                    span[2] = indx

                if indx == len(seq) - 1:
                    spans.append(span)
            else:
                if span[2] != -1:
                    spans.append(span)
                span = [-1, -1, -1]
        return spans
