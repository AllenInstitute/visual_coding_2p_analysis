import os

from manifest import prepare_manifest, test_json_files
from allensdk.core.brain_observatory_cache import BrainObservatoryCache


def get_boc_json_files(manifest_file):
    # We need 'base_uri' until all of the data is publicly released
    boc = BrainObservatoryCache(manifest_file=manifest_file, base_uri="http://swarehouse/")

    exps = boc.get_ophys_experiments()
    cells = boc.get_cell_specimens()
    ecs = boc.get_experiment_containers()


def get_boc_data_files(manifest_file):
    # We need 'base_uri' until all of the data is publicly released
    boc = BrainObservatoryCache(manifest_file=manifest_file, base_uri="http://swarehouse/")

    exps = boc.get_ophys_experiments()
    cells = boc.get_cell_specimens()
    ecs = boc.get_experiment_containers()

    for i, exp in enumerate(exps):
        print("%d %d/%d" % (exp['id'], i+1, len(exps)))
        boc.get_ophys_experiment_data(exp['id'])
        boc.get_ophys_experiment_events(exp['id'])
        # boc.get_ophys_experiment_analysis(ophys_experiment_id=exp['id'],
        #                                   stimulus_type='natural_movie_one')


def count_cells(manifest_file):
    # We need 'base_uri' until all of the data is publicly released
    boc = BrainObservatoryCache(manifest_file=manifest_file, base_uri="http://swarehouse/")
    exps = boc.get_ophys_experiments()

    total_cells = 0
    for i, exp in enumerate(exps):
        data = boc.get_ophys_experiment_data(exp['id'])
        time, dff = data.get_dff_traces()
        N, _ = dff.shape
        total_cells += N

    print "Found ", total_cells, " cells in ophys data files."
        

if __name__ == '__main__':

    manifest_dir = '/allen/programs/braintv/workgroups/cortexmodels/michaelbu/ObservatoryPlatformPaperAnalysis/platform_boc_2018_09_25'
    manifest_file = os.path.join(manifest_dir, 'manifest.json')

    get_boc_json_files(manifest_file)
    prepare_manifest(manifest_dir)
    get_boc_data_files(manifest_file)
    count_cells(manifest_file)
    test_json_files(manifest_file)

    



