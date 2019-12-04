import sys

from filepath.filepath import fp

from scapy.all import *

from kabab.config import CLASSES


def get_label(mpath):

    idx = 0
    res = None

    for c in CLASSES:
        if c in mpath:
            res = idx
        idx += 1

    if res is None:
        raise Exception("Unknown label {}".format(mpath))

    return res


def read_inputs():
    # parse input
    if len(sys.argv) < 2:
        print_help()
        exit(1)
    # parse args
    arg_dict = dict(list(enumerate(sys.argv)))
    data_dir = fp(sys.argv[1])  # ARG 1 (required) input dir
    out_file = arg_dict.get(2, "output.csv")  # ARG 2 (optional) output file (default: output.csv)
    return data_dir, out_file


def print_help():
    pass


def get_pcaps(data_dir):
    pcap_list = list([p for p in data_dir.find_files() if p.ext() not in ['json', 'csv', 'txt', 'data']])
    return pcap_list


def load_pcap(pf):
    """
    Read pcap file into Scapy Ether objects
    :return: File path - Ether object tuples
    """
    fpath = str(pf)
    return fpath, rdpcap(fpath)