from visual_coding_2p_analysis.l0_analysis import L0_analysis

from allensdk.core.brain_observatory_cache import BrainObservatoryCache

MANIFEST_FILE = '/allen/aibs/technology/allensdk_data/platform_boc_pre_2018_3_16/manifest.json'

def generate_events(n):

    boc = BrainObservatoryCache(manifest_file=MANIFEST_FILE)

    expt_list = boc.get_ophys_experiments()
    total = len(expt_list)
    batch_size = total/40 + (1 if total%40!=0 else 0)

    lower = batch_size*n
    upper = lower + batch_size

    for expt in expt_list[lower:upper]:

        eid = expt['id']
        print "Processing events for ", eid
        data = boc.get_ophys_experiment_data(eid)
        l0 = L0_analysis(data)

        events = l0.get_events()

    print "Done"


def generate_events_for_expt(eid):

    boc = BrainObservatoryCache(manifest_file=MANIFEST_FILE)

    print "Processing events for ", eid
    data = boc.get_ophys_experiment_data(eid)
    l0 = L0_analysis(data)

    events = l0.get_events()

    


if __name__ == '__main__':

    import sys

    n = int(sys.argv[1])

    # generate_events(n)

    generate_events_for_expt(n)