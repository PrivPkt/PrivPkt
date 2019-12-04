import numpy as np

from filepath.filepath import fp
from pyspark.sql.types import Row
from pyspark.taskcontext import TaskContext
from scipy.io import savemat
from scapy.all import *
from scapy.all import DNS, TCP, IPv6, IP, UDP

from kabab.flow import explode_pcap_to_packets
from kabab.utils.general import get_label, read_inputs, get_pcaps, load_pcap
from kabab.config import PARTITIONS
from kabab.utils.sprk import get_spark_session

import config as kconfig


def filter_out_dns(pkt):
    return DNS not in pkt


def filter_out_empty_tcp(pkt):
    if TCP not in pkt:
        return True
    return len(pkt.getlayer(TCP).payload) > 0


def filter_out_non_ip(pkt):
    return IP in pkt or IPv6 in pkt


def filter_out_irrelavent(arg):
    _, _, pkt = arg
    return filter_out_dns(pkt) and \
        filter_out_empty_tcp(pkt) and \
        filter_out_non_ip(pkt)


def _get_ip_layer(pkt):
    return pkt.getlayer(IP) if IP in pkt \
        else pkt.getlayer(IPv6)


def _get_zero_address(pkt):
    return '0.0.0.0' if IP in pkt \
        else '::1'


def convert_to_bytes(arg):
    f, i, pkt = arg

    assert IP in pkt or IPv6 in pkt

    # get layer-3
    netlayer = _get_ip_layer(pkt).copy()

    # mask IP
    if kconfig.MASK_IP:
        zero = _get_zero_address(pkt)
        netlayer.src = zero
        netlayer.dst = zero

    # add IP header
    header_length = len(netlayer) - len(netlayer.payload)
    mbytes = str(netlayer)[:header_length]

    # mask TCP/UDP port
    if kconfig.MASK_PORT:
        if UDP in pkt or TCP in pkt:
            netlayer.sport = 0
            netlayer.dport = 0

    # zero-pad UDP header
    if UDP in pkt:
        mbytes += str(netlayer.getlayer(UDP))[:8]
        mbytes += '\0' * 12
        mbytes += str(netlayer.getlayer(UDP))[8:]
    else:
        mbytes += str(netlayer.payload)

    # use first 1500 bytes
    mbytes = mbytes[:1500]

    # TODO remove this madness and use byte-arrays
    # convert to integer values and zero-pad
    mbytes = [ord(c) for c in mbytes]
    mbytes += max(1500 - len(mbytes), 0) * [0]

    assert len(mbytes) == 1500

    return f, i, mbytes


def to_row(arg):

    fpath, idx, mbytes = arg
    assert len(mbytes) == 1500
    for c in mbytes:
        assert type(c) == int and 0 <= c <= 255

    mdict = {"label": (get_label(fpath))}

    for i in range(1500):
        mdict["b%03d" % i] = mbytes[i]

    return Row(**mdict)


def _save_mat_f(outfile):

    def _save_mat(partition):

        ctx = TaskContext()
        data = []

        for row in partition:
            fpath, idx, mbytes = row
            for b in mbytes:
                assert isinstance(b, int)
            row_data = mbytes + [get_label(fpath)]
            data.append(row_data)

        mat = np.array(data, dtype=np.dtype('B'))
        out = fp(outfile) + fp("part_%05d.mat" % ctx.partitionId())
        savemat(out.path(), {"packets": mat})

    return _save_mat


def deep_packet(use_mat=False):

    # get input and output dir
    data_dir, out_file = read_inputs()

    # list PCAP files
    pcap_list = get_pcaps(data_dir)

    spark, sc, sqlContext = get_spark_session()

    # make RDD
    paths_rdd = sc.parallelize(pcap_list)

    analyzed_rdd = paths_rdd \
        .repartition(PARTITIONS) \
        .map(load_pcap) \
        .flatMap(explode_pcap_to_packets) \
        .filter(filter_out_irrelavent) \
        .map(convert_to_bytes)

    if not use_mat:
        analyzed_rdd \
            .repartition(PARTITIONS) \
            .map(to_row) \
            .toDF() \
            .coalesce(1) \
            .write \
            .csv(out_file, header=True)
    else:

        NUM_MAT_FILES = 2 ** 14

        fp(out_file).ensure()
        abspath = os.path.abspath(fp(out_file).path())

        analyzed_rdd \
            .map(lambda x: (0, x)) \
            .partitionBy(NUM_MAT_FILES, lambda x: np.random.randint(NUM_MAT_FILES)) \
            .map(lambda x: x[1]) \
            .foreachPartition(_save_mat_f(abspath))


def analysis():

    global psizes, res

    # get input and output dir
    data_dir, out_dir = read_inputs()

    # list PCAP files
    pcap_list = get_pcaps(data_dir)

    spark, sc, sqlContext = get_spark_session()

    # make RDD
    paths_rdd = sc.parallelize(pcap_list)

    psizes = paths_rdd \
        .repartition(PARTITIONS) \
        .map(load_pcap) \
        .flatMap(explode_pcap_to_packets) \
        .filter(filter_out_irrelavent) \
        .map(lambda s: len(s[2].payload))

    res = psizes.collect()

    psizes.persist()\
        .map(lambda s: Row(size=s))\
        .toDF()\
        .createOrReplaceTempView("psizes")

    spark.sql("""
        select 
            size, 
            count(size) as cnt
        from psizes
        group by size
        order by size
    """) \
        .coalesce(1)\
        .write\
        .csv(out_dir + "/hist-1.csv", header=True)


def __main__():
    deep_packet(use_mat=True)


if __name__ == "__main__":
    __main__()


