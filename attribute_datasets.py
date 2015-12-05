"""
Module provides iterator functions for the PARSE-27k dataset.
"""
import numpy as np
import os, re
from itertools import izip
from collections import OrderedDict
import sqlite3

class PersonExample(object):
    ''' Manage one annotated pedestrian example
    - all labels are in 'softmax' notation, i.e. (0..N-1) for N classes,
    where 0 is the NA label.
    - This class provides the functionality to 'mirror' labels
    '''
    def __init__(self, attributes, valid_values,
                 pid, image_filename, box, labels):
            self.attributes = attributes
            self.valid_values = valid_values
            self.pid = pid
            self.image_filename = image_filename
            self.box = box
            self._labels = OrderedDict(zip(self.attributes, labels))

    def label(self, attr_idx, mirrored=False):
        return self.labels(mirrored=mirrored)[attr_idx]

    def labels(self, mirrored=False):
        # mirror the labels for this example
        if not mirrored:
            return self._labels.values() # _labels is an ordered dict
        else:
            labels_out = []
            binary_mirror_attributes = {
                          'HasBagInHandLeft'     : 'HasBagInHandRight',
                          'HasBagInHandRight'    : 'HasBagInHandLeft',
                          'HasBagOnShoulderLeft' : 'HasBagOnShoulderRight',
                          'HasBagOnShoulderRight': 'HasBagOnShoulderLeft' }
            for attr, value in self._labels.items():
                if attr == 'Orientation12':
                    raise ValueError('we cannot mirror Ori12 yet')
                elif attr == 'Orientation4':
                    # input in softmax notation:
                    # 0 na, 1 front, 2 back, 3 left, 4 right
                    # translation permutation: (0)(1)(2) (3,4)
                    table = {0:0, 1:1, 2:2,
                             3:4, 4:3}
                    try:
                        out = table[value]
                    except KeyError:
                        msg = 'invalid Orientation label in softmax notation: {}'
                        raise ValueError(msg.format(value))
                    labels_out.append(out)
                elif attr == 'Orientation8':
                    # input in softmax notation:
                    # 0 na, 1 front, 2 front-right, 3 right
                    # 4 back-right, 5 back, 6 back-left, 7 left, 8 front-left
                    # translation permutation: (0)(1)(2,8)(3,7)(4,6)(5)
                    table = {0:0, 1:1,
                             2:8, 8:2,
                             3:7, 7:3,
                             4:6, 6:4,
                             5:5}
                    try:
                        out = table[value]
                    except KeyError:
                        msg = 'invalid Orientation8 label in softmax notation: {}'
                        raise ValueError(msg.format(value))
                    labels_out.append(out)
                elif attr in binary_mirror_attributes.keys():
                    labels_out.append(self._labels[binary_mirror_attributes[attr]])
                else:
                    labels_out.append(value) # no change for mirrored example
            return labels_out

    def _one_hot(self, idx, max_n):
        z = np.zeros(max_n)
        z[idx] = 1
        return z

    def label_one_hot(self, attr_idx=None, mirrored=False):
        label = self.label(attr_idx=attr_idx, mirrored=mirrored)
        return self._one_hot(label, self.valid_values[attr_idx])

    def labels_one_hot(self, attr_idx=None, mirrored=False):
        labels = self.labels(mirrored)
        return [ self._one_hot(l, self.valid_values[idx]) for idx,l in enumerate(labels) ]

class DataPreprocessorPARSE(object):
    def __init__(self, pathToDataset,
                 fnAnnoDB='annotations.sqlite3',
                 attributes='all',
                 split='train',
                 exclude_sitting=True,
                 verbose=False):
        """reads annotation from sqlite3 db at fnAnno, and assumes images to be
        at pathToDataset
        - pathToDataset path to sequence directories containing the images
        - fnAnnoDB      sqlite3 annotation database
        - attributes    list of attributes to return
        - split         which examples to iterate over (train,val,test)
        - exclude_sitting - if True treat Posture as binary (NA, walking, standing)
            (there are only very few sitting examples, and we did not use these in our work)
        - verbose - show SQL query
        """
        self.__dict__.update(locals())
        self.pathToSequences = os.path.join(pathToDataset, 'sequences')

        self.ALL_ATTRIBUTES = ('Orientation',
                               'Orientation8',
                               'Gender',
                               'Posture',
                               'HasBagOnShoulderLeft', 'HasBagOnShoulderRight',
                               'HasBagInHandLeft', 'HasBagInHandRight',
                               'HasTrolley',
                               'HasBackpack',
                               'isPushing',
                               'isTalkingOnPhone')
        if not self.exclude_sitting:
            self.VALID_VALUES = [5,9,3,4,3,3,3,3,3,3,3,3] # max N per item in ALL_ATTRIBUTES
        else:
            self.VALID_VALUES = [5,9,3,3,3,3,3,3,3,3,3,3] # max N per item in ALL_ATTRIBUTES

        if isinstance(attributes, str) and attributes.lower() == 'all':
            self.attributes = self.ALL_ATTRIBUTES
            self.valid_values = self.VALID_VALUES
        elif isinstance(attributes, list) and len(attributes) > 0:
            self.attributes = attributes
            self.valid_values = [self.VALID_VALUES[self.ALL_ATTRIBUTES.index(a)] for a in self.attributes]
        else:
            raise ValueError('invalid attribute selection')

        self.TRAIN_SEQUENCES = (1, 4, 5)
        self.VAL_SEQUENCES = (2, 7, 8)
        self.TEST_SEQUENCES = (3, 6)
        self.TRAINVAL_SEQUENCES = (1, 4, 5, 2, 7, 8)
        self.ALL_SEQUENCES = (1, 2, 3, 4, 5, 6, 7, 8)

        if split == 'train':
            self.sequenceIDs = self.TRAIN_SEQUENCES
        elif split == 'val':
            self.sequenceIDs = self.VAL_SEQUENCES
        elif split == 'test':
            self.sequenceIDs = self.TEST_SEQUENCES
        else:
            raise ValueError('invalid split')

        self.examples = self._read_examples_from_db()

    def _read_examples_from_db(self):
        try:
            self.dbFile = os.path.join(self.pathToDataset, self.fnAnnoDB)
            self.db = sqlite3.connect(self.dbFile)
            self.dbc = self.db.cursor()
        except sqlite3.Error as e:
            raise Exception(e)

        query = '''
            SELECT s.directory as directory,
                   i.filename as filename,
                   p.pedestrianID as pid,
                   p.box_min_x as min_x, p.box_min_y as min_y,
                   p.box_max_x as max_x, p.box_max_y as max_y,
                   {0}
            FROM Pedestrian p
            INNER JOIN AttributeSet a ON p.attributeSetID = a.attributeSetID
            INNER JOIN Image i ON p.imageID = i.imageID
            INNER JOIN Sequence s on s.sequenceID = i.sequenceID
        '''.format(', '.join((a+'ID' for a in self.attributes)))
        if self.exclude_sitting:
            query += ' WHERE a.postureID <> 4 ' # filter out all 'sitting' examples
            if self.sequenceIDs:
                query += 'AND i.sequenceID IN ' + str(self.sequenceIDs)
        else:
            if self.sequenceIDs:
                query += ' WHERE i.sequenceID IN ' + str(self.sequenceIDs)
        if self.verbose:
            print(query)

        results = self.dbc.execute(query).fetchall()
        examples = []
        for row in results:
            # tuples are: (PedestrianID, ... all the attributes)
            fullFileName = os.path.join(self.pathToSequences, row[0], row[1])
            box = tuple(row[3:7])
            pid = str(row[2])
            labels = self._translate_db_labels_to_softmax(row[7:])
            p = PersonExample(self.attributes, self.valid_values,
                              pid, fullFileName.encode('utf-8'),
                              box, labels)
            examples.append(p)
        self.db.close()
        return examples

    def _translate_db_label_to_softmax(self, attribute, label):
        """
        translates a label from the sqlite database to a softmax label.
        The softmax range is (0,1,...,N) - where 0 is the N/A label.
        (0=NA, 1=POS, 2=NEG)
        (0=NA, 1=front, 2=back, 3=left, 4=right)
        """
        msg = 'unexpected label - attribute: {} - value: {}'
        if not isinstance(label, int):
            raise TypeError('label expected to be integer')
        if not attribute in self.attributes:
            raise ValueError('invalid attribute')

        # translate to range [0,1,..N]
        # by convention we handled the male as the 'pos' label
        # this can have an influence on the exact value of AP scores
        if attribute == 'Posture':
            if self.exclude_sitting: # if we handle Posture as binary (the default)
                if label == 3: # standing -> pos (less frequent)
                    out = 1
                elif label == 2: #walking -> neg (the more frequent class)
                    out = 2
                elif label == 1:
                    out = 0
                else:
                    raise ValueError(msg.format(attribute, label))
        else:
            out = label - 1
        return out

    def _translate_db_labels_to_softmax(self, labels):
        """
        applies translation all attributes
        - should be useful when we support working only on a subset of
         attributes
        """
        out_labels = []
        if len(self.attributes) != len(labels):
            msg = 'length of labels does not match my attribute count!'
            raise ValueError(msg)

        out_labels = [self._translate_db_label_to_softmax(a, l) for (a, l)
                      in izip(self.attributes, labels)]
        return out_labels

    @property
    def all_examples(self):
        return list(self.examples) # return a copy of our internal list

    def __iter__(self):
        """iterate over all annotation examples
        """
        for example in self.examples:
            yield example
