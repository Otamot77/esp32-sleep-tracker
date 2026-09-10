from sleep_server.protocol import PacketParser, PpgPacket, ImuPacket, encode_ppg, encode_imu

def test_ppg_round_trip():
    raw = encode_ppg(42, 123456789, 100001, 120002)
    packets = PacketParser().feed(raw)
    assert len(packets) == 1
    p = packets[0]
    assert isinstance(p, PpgPacket)
    assert (p.sequence, p.device_time_us, p.red, p.ir) == (42, 123456789, 100001, 120002)

def test_imu_round_trip():
    raw = encode_imu(7, 500000, 0.1, 0.2, 9.8, 0.01, 0.02, 0.03, 31.2)
    p = PacketParser().feed(raw)[0]
    assert isinstance(p, ImuPacket)
    assert p.sequence == 7
    assert abs(p.az - 9.8) < 1e-5

def test_fragmented_tcp_stream():
    raw = encode_ppg(1, 1000, 10, 20)
    parser = PacketParser()
    assert parser.feed(raw[:5]) == []
    assert parser.feed(raw[5:12]) == []
    assert len(parser.feed(raw[12:])) == 1

def test_multiple_packets_in_one_read():
    raw = encode_ppg(1, 1000, 10, 20) + encode_ppg(2, 2000, 11, 21)
    packets = PacketParser().feed(raw)
    assert [p.sequence for p in packets] == [1, 2]
