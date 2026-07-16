import utils


def rank(queries, index, k1=0.9, b=0.4, count=1000):
    baseline = f"-baseline=okapi,k1:{k1},b:{b},k3:{count}"
    return utils.indri_run(queries, index, [baseline], count)
