import logging
import struct


int_size = 4


def _retrieve_buffer(conn, length):
    data = bytearray()

    while length > 0:
        pkt = conn.recv(length)
        if len(pkt) == 0:
            logging.info("Frontend disconnected.")
            raise ValueError("Frontend disconnected")

        data += pkt
        length -= len(pkt)

    return data

def _retrieve_int(conn):
    data = _retrieve_buffer(conn, int_size)
    return struct.unpack("!i", data)[0]

def retrieve_request(conn):
    content_len = _retrieve_int(conn)
    content = _retrieve_buffer(conn, content_len)
    return content