"""Reproduce 208 training-only validation experiments, without reading test files.

The BN's four manually fixed three-node chains are never searched or learned.
This entry point keeps large intermediate feature arrays in --output-dir.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from experiments import explore as e
from experiments import shape_round as s


def source_hashes():
    root=Path(__file__).resolve().parent
    return {name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
            ('optimize_target90.py','experiments/explore.py','experiments/shape_round.py')}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,default=e.OUT)
    args=parser.parse_args()
    e.OUT=args.output_dir
    e.OUT.mkdir(parents=True,exist_ok=True)
    X,y=e.previous.load_training_only(args.data_dir)
    train,valid=e.previous.validation_split(y)
    rows=e.nb_round(X,y,train,valid)+e.bn_round(X,y,train,valid)+s.run_shapes(X,y,train,valid)
    assert len(rows)==208
    best={key:max((r for r in rows if r['model']==key),key=lambda r:(r['macro_f1'],r['accuracy']))
          for key in ('naive_bayes','bayes_net')}
    selected_bn=best['bayes_net']
    values=np.load(e.OUT/(selected_bn['features']+'.npy'))
    states=e.previous.discretize_features(values,np.asarray(selected_bn['edges']))
    no_edges=e.previous.scores_from_features(states[train],y[train],states[valid],selected_bn['alpha'],(None,)*12).argmax(1)
    result={'seed':20261019,'training_samples':len(train),'validation_samples':len(valid),
            'validation_class_counts':np.bincount(y[valid],minlength=10).tolist(),
            'ranking':'macro_f1, then accuracy','test_files_read':False,
            'automatic_structure_learning':False,'manual_feature_parents_zero_based':list(e.PARENTS),
            'candidate_count':len(rows),'candidates':rows,'selected':best,
            'selected_bn_no_feature_edges_ablation':e.previous.metrics(y[valid],no_edges),
            'source_sha256':source_hashes()}
    (e.OUT/'selection.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    np.savetxt(e.OUT/'validation_indices.csv',valid,fmt='%d',header='training_set_index',comments='')
    print('SELECTED',json.dumps(best),flush=True)


if __name__=='__main__':
    main()
