"""
End-to-End 3-Tier Integration & Sub-Millisecond Benchmark Test
"""

import time
import numpy as np

from crypto_engine import encrypt_data, decrypt_data
from stego_engine import get_cover_matrix, embed_payload_into_matrix, extract_payload_from_array
from lstm_traffic_engine import inspect_traffic_sequence


def run_pipeline_benchmark(iterations=200):
    print("=" * 70)
    print(f"🚀 RUNNING SUB-MILLISECOND 3-TIER DEFENSE BENCHMARK ({iterations} RUNS)")
    print("=" * 70)

    raw_telemetry = b"TELEMETRY: Bus=14, Voltage=230.15V, Frequency=50.02Hz, Status=OPTIMAL"
    cover = get_cover_matrix("sample_cover.png")
    
    np.random.seed(42)
    test_traffic_seq = np.random.normal(0.20, 0.05, (10, 41))

    t_enc_list, t_emb_list, t_ids_list, t_ext_list, t_dec_list = [], [], [], [], []
    is_clean = False
    recovered = b""

    # Warm up caches
    for _ in range(10):
        _c = encrypt_data(raw_telemetry)
        _s = embed_payload_into_matrix(cover, _c)
        _x = extract_payload_from_array(_s)
        _ = decrypt_data(_x)
        _ = inspect_traffic_sequence(test_traffic_seq)

    for _ in range(iterations):
        # Layer 1: AES-256 Encryption
        t0 = time.perf_counter()
        cipher = encrypt_data(raw_telemetry)
        t_enc_list.append((time.perf_counter() - t0) * 1000)

        # Layer 2: Stego Embedding
        t0 = time.perf_counter()
        stego_matrix = embed_payload_into_matrix(cover, cipher)
        t_emb_list.append((time.perf_counter() - t0) * 1000)

        # Layer 3: CNN-LSTM AI Threat Inspection
        t0 = time.perf_counter()
        is_clean = inspect_traffic_sequence(test_traffic_seq)
        t_ids_list.append((time.perf_counter() - t0) * 1000)

        # Layer 2: Stego Extraction
        t0 = time.perf_counter()
        extracted_cipher = extract_payload_from_array(stego_matrix)
        t_ext_list.append((time.perf_counter() - t0) * 1000)

        # Layer 1: AES-256 Decryption
        t0 = time.perf_counter()
        recovered = decrypt_data(extracted_cipher)
        t_dec_list.append((time.perf_counter() - t0) * 1000)

    avg_enc = np.mean(t_enc_list)
    avg_emb = np.mean(t_emb_list)
    avg_ids = np.mean(t_ids_list)
    avg_ext = np.mean(t_ext_list)
    avg_dec = np.mean(t_dec_list)
    total_avg = avg_enc + avg_emb + avg_ids + avg_ext + avg_dec

    assert is_clean is True, "AI Engine False Positive!"
    assert recovered == raw_telemetry, "Data Integrity Mismatch!"

    print("✔ Data Integrity Test : 100% RECOVERED (Zero Bit Loss)")
    print("✔ AI Threat Detector  : VERIFIED (Traffic correctly classified as CLEAN)")
    print("-" * 70)
    print(f"1. AES-256 Encryption Latency : {avg_enc:.4f} ms")
    print(f"2. Stego Embedding Latency    : {avg_emb:.4f} ms")
    print(f"3. CNN-LSTM Threat Check      : {avg_ids:.4f} ms")
    print(f"4. Stego Extraction Latency   : {avg_ext:.4f} ms")
    print(f"5. AES-256 Decryption Latency : {avg_dec:.4f} ms")
    print("-" * 70)
    print(f"🔥 TOTAL END-TO-END LATENCY   : {total_avg:.4f} ms")
    print(f"STATUS                        : {'PASS (< 1.0 ms) 🚀' if total_avg < 1.0 else 'FAIL (> 1.0 ms)'}")
    print("=" * 70)


if __name__ == "__main__":
    run_pipeline_benchmark()