from kabab.packet import check_packet_tls, get_src_dst
from kabab.utils import get_logger


# logger is parse

logger = get_logger("Parse")


def get_base_pkt(cap):
    return cap[0]


def check_tls(pcap):

    has_tls, has_h1, has_h2 = False, False, False

    for pkt in pcap:

        a = check_packet_tls(pkt)

        has_tls = has_tls or a["has_tls"]
        has_h1 = has_h1 or a["has_h1"]
        has_h2 = has_h2 or a["has_h2"]

    return has_tls, has_h1, has_h2


def inter_arrival(pcap):

    previous = None

    for current in pcap:
        if previous:
            yield current.time - previous.time
        previous = current


def size_seq(pcap):

    for pkt in pcap:
        yield len(pkt.payload)


def dir_seq(pcap):

    base_src_dst = get_src_dst(get_base_pkt(pcap))

    return (1 if get_src_dst(pkt) == base_src_dst else -1 for pkt in pcap)


def explode_pcap_to_packets(arg):

    fpath, pcap = arg

    for idx, pkt in enumerate(pcap):
        yield fpath, idx, pkt



