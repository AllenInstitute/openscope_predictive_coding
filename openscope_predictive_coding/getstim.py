from openscope_predictive_coding.stimtable import get_stimtable
from intervaltree import Interval, IntervalTree
from openscope_predictive_coding.utilities import memoized

@memoized
def get_interval_tree(session):
    
    df = get_stimtable(session)

    interval_tree = IntervalTree()
    for ii, row in df.iterrows():
        row_dict = row.to_dict()
        start, end = row_dict.pop('start'), row_dict.pop('end')
        interval_tree[start:end] = row_dict

    return interval_tree

def get_stimulus(t, session):
    return get_interval_tree(session)[t]

if __name__ == '__main__':

    print get_stimulus(500., 'A')