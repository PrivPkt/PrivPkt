from scapy.all import *
from scapy.all import TCP, UDP, IP, IPv6
from scapy_ssl_tls.ssl_tls import TLS, TLSServerHello, TLSClientHello

from kabab.utils.gen import pick_first_n


def _get_ip_src_dst(pkt):
    # layer = IPv6 if IPv6 in pkt else IP
    return pkt.src, pkt.dst


def _get_tcp_udp_src_dst(pkt, delimiter=":"):
    """
    :param pkt: Scapy frame
    :param layer: TCP or UDP
    :return:
    """
    ip = _get_ip_src_dst(pkt)
    return ip[0] + delimiter + str(pkt.sport), ip[1] + delimiter + str(pkt.dport)


def get_first_n_bytes(pcap, n=784):
    return pick_first_n((len(pkt.load) for pkt in pcap), n=n)


def check_packet_tls(pkt):

    has_tls, has_h1, has_h2 = False, False, False

    if TLS in pkt:
        has_tls = True

    if TLSServerHello in pkt:
        for protocol in extract_next_protocol(pkt):
            if re.match(r"http/1.*", protocol):
                has_h1 = True
            if "h2" in protocol:
                has_h2 = True

    return {
        "has_tls": has_tls,
        "has_h1": has_h1,
        "has_h2": has_h2
    }


def get_src_dst(pkt):

    if TCP in pkt or UDP in pkt:
        return _get_tcp_udp_src_dst(pkt)
    elif IP in pkt or IPv6 in pkt:
        return _get_ip_src_dst(pkt)
    else:
        return pkt.src, pkt.dst


def _add_dict(d1, d2):
    d = dict(d1)
    d.update(d2)
    return d


def apply_function_to_pkt(function):
    """
    returns a function
    :param function:
    :return:
    """
    return lambda pkt_tuple: (
        pkt_tuple[0],
        pkt_tuple[1],
        _add_dict(pkt_tuple[2], function(pkt_tuple[1]))
    )


def extract_next_protocol(pkt):

    if TLSServerHello in pkt:
        layer = TLSServerHello
    elif TLSClientHello in pkt:
        layer = TLSClientHello
    else:
        raise Exception("No ALPN data available")

    protocols = []

    for extension in pkt.getlayer(layer).extensions:
        if extension.type == 16:
            for protocol in extension.protocol_name_list:
                protocols.append(protocol.data)

    return protocols