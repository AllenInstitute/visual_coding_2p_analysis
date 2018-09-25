import os
import json

from allensdk.core.brain_observatory_cache import BrainObservatoryCache


PV = 'Pvalb-IRES-Cre'
GC6S = 'Slc17a7-IRES2-Cre/wt;Camk2a-tTA/wt;Ai94(TITL-GCaMP6s)/wt'


def load_json_file(json_file):
    with open(json_file) as f:
        return json.load(f)

def save_json_file(obj, json_file):
    with open(json_file, 'w') as f:
        json.dump(obj, f)

def filter_out_pv_6s(exp_or_cont):

    genotype = lambda x: x['specimen']['donor']['full_genotype']
    cre_line = lambda x: genotype(x).split('/')[0]

    no_pv_exps = [exp for exp in exp_or_cont if cre_line(exp)!=PV]
    good_exps = [exp for exp in no_pv_exps if genotype(exp)!=GC6S]

    return good_exps

def filter_out_pv_6s_for_cells(cells):

    genotype = lambda x:  x['donor_full_genotype']
    cre_line = lambda x: genotype(x).split('/')[0]

    no_pv_cells = [cell for cell in cells if cre_line(cell)!=PV]
    good_cells = [cell for cell in no_pv_cells if genotype(cell)!=GC6S]

    return good_cells


def filter_json_file(json_file, filter_to_use):
    json_data = load_json_file(json_file)
    good = filter_to_use(json_data)
    save_json_file(good, json_file)


def prepare_manifest(manifest_dir):

    ophys_file = os.path.join(manifest_dir, 'ophys_experiments.json')
    cont_file = os.path.join(manifest_dir, 'experiment_containers.json')
    cell_file = os.path.join(manifest_dir, 'cell_specimens.json')

    filter_json_file(ophys_file, filter_to_use=filter_out_pv_6s)
    filter_json_file(cont_file, filter_to_use=filter_out_pv_6s)
    filter_json_file(cell_file, filter_to_use=filter_out_pv_6s_for_cells)


def test_json_files(manifest_file):

    boc = BrainObservatoryCache(manifest_file=manifest_file, base_uri="http://swarehouse/")

    exps = boc.get_ophys_experiments()
    assert len(exps)==1296, "Wrong number of ophys experiments.  Found "+str(len(exps))
    print "Passed:  ", len(exps), " experiments found (expecting 1296)."

    conts = boc.get_experiment_containers()
    assert len(conts)==432, "Wrong number of experiment containers.  Found "+str(len(conts))
    print "Passed:  ", len(conts), " experiment containers found (expecting 432)."

    cells = boc.get_cell_specimens()
    assert len(cells)==61371, "Wrong number of cell specimens.  Found "+str(len(cells))
    print "Passed:  ", len(cells), " cell specimens found (expecting 61371)."

if __name__ == '__main__':

    MANIFEST_DIR = os.path.join(os.path.dirname(__file__),'data')
    MANIFEST_FILE = os.path.join(MANIFEST_DIR, 'manifest.json')

    prepare_manifest(MANIFEST_DIR)
    test_json_files(MANIFEST_FILE)

    
    

