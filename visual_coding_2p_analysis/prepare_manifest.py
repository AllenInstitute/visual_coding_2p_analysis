import os


from allensdk.core.brain_observatory_cache import BrainObservatoryCache

TEMP_MANIFEST_FILE = os.path.join(os.path.dirname(__file__), 'data/manifest.json')

# print TEMP_MANIFEST_FILE
# import sys
# sys.exit()

def get_boc_json_files():
    # We need 'base_uri' until all of the data is publicly released
    boc = BrainObservatoryCache(manifest_file=TEMP_MANIFEST_FILE, base_uri="http://swarehouse/")

    exps = boc.get_ophys_experiments()
    cells = boc.get_cell_specimens()
    ecs = boc.get_experiment_containers()


    print boc.get_all_cre_lines()

# for i, exp in enumerate(exps):
#     print("%d %d/%d" % (exp['id'], i+1, len(exps)))
#     boc.get_ophys_experiment_data(exp['id'])
#     boc.get_ophys_experiment_events(exp['id'])
#     boc.get_ophys_experiment_analysis(ophys_experiment_id=exp['id'],
#                                       stimulus_type='natural_movie_one')



if __name__ == '__main__':

    get_boc_json_files()