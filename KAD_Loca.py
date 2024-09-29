import matplotlib
matplotlib.use('TkAgg')


import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Define the nodes and their coordinates
nodes = {
    'China': (35.8617, 104.1954),
    'UnitedStates': (37.0902, -95.7129),
    'India': (20.5937, 78.9629),
    'Brazil': (-14.235, -51.9253),
    'Russia': (61.5240, 105.3188),
    'Australia': (-25.2744, 133.7751),
    'SouthAfrica': (-30.5595, 22.9375),
    'Canada': (56.1304, -106.3468),
    'Mexico': (23.6345, -102.5528),
    'Argentina': (-38.4161, -63.6167),
    'USA2': (39.8283, -98.5795),
    'Canada2': (56.1304, -106.3468),
    'Germany': (51.1657, 10.4515),
    'France': (46.6031, 1.8206),
    'Italy': (41.8719, 12.5674),
    'Spain': (40.4637, -3.7492),
    'UnitedKingdom': (55.3781, -3.4360),
    'Japan': (36.2048, 138.2529),
    'SouthKorea': (35.9078, 127.7669),
    'India2': (20.5937, 78.9629),
    'China2': (35.8617, 104.1954),
    'Australia2': (-25.2744, 133.7751),
    'NewZealand': (-40.9006, 174.8860),
    'Fiji': (-17.7134, 178.0650),
    'PapuaNewGuinea': (-6.3150, 143.9555),
    'SouthAfrica2': (-30.5595, 22.9375),
    'Nigeria': (9.0820, 8.6753),
    'Kenya': (-1.2921, 36.8219),
    'Egypt': (26.8206, 30.8025),
}

import random
import time
from collections import defaultdict
from dataclasses import dataclass

KAD_PORT = 8000  # Set appropriate port
KAD_ID_LEN = 160

@dataclass
class Block:
    blockID: int
    # Other properties can be added as needed

@dataclass
class Chunk:
    blockID: int
    chunkID: int
    prevID: int
    blockSize: int
    chunkSize: int
    nChunks: int

class KadcastNode:
    kadK = 20
    kadAlpha = 3
    kadBeta = 3
    kadFecOverhead = 0.25

    def __init__(self, address, is_miner, hash_rate):
        self.address = address
        self.is_miner = is_miner
        self.hash_rate = hash_rate
        self.nodeID = self.generate_node_id()
        self.done_blocks = [True] * 10  # Adjust size as needed
        self.buckets = defaultdict(list)  # Bucket structure
        self.active_buckets = set()
        self.pending_refreshes = {}
        self.send_queue = []
        self.sending = False
        self.is_running = False
        self.max_seen_height = {}

    def start_application(self):
        print(f"Starting node {self.nodeID} at address {self.address}")
        self.is_running = True
        self.bootstrap_peers()
        if self.is_miner:
            time.sleep(200)
            self.start_mining()

    def stop_application(self):
        print("Stopping application")
        self.is_running = False
        self.pending_refreshes.clear()

    def generate_node_id(self):
        return self.random_id_in_interval(0, 2**KAD_ID_LEN)

    def random_id_in_interval(self, min_id, max_id):
        return random.randint(min_id, max_id)

    def bootstrap_peers(self):
        # Bootstrap logic for known addresses
        for addr in self.known_addresses:
            if addr != self.address:
                time.sleep(random.gauss(10, 5))  # Simulate delay
                self.send_ping_message(addr)

    def update_bucket(self, addr, node_id):
        if node_id == self.nodeID:
            return
        i = self.bucket_index_from_id(node_id)
        self.active_buckets.add(i)

        if len(self.buckets[i]) < self.kadK:
            self.buckets[i].append((addr, node_id))
        else:
            self.buckets[i].pop(0)  # Remove the oldest
            self.buckets[i].append((addr, node_id))

    def init_broadcast(self, block):
        self.done_blocks[block.blockID] = True
        if block.blockID not in self.max_seen_height:
            self.max_seen_height[block.blockID] = KAD_ID_LEN
        self.broadcast_block(block)

    def broadcast_block(self, block):
        height = self.max_seen_height.pop(block.blockID)
        if height == 0:
            return

        for b_index in range(height - 1, -1, -1):
            if not self.buckets[b_index]:
                continue
            node_addresses = random.sample(self.buckets[b_index], min(self.kadBeta, len(self.buckets[b_index])))
            for n_addr in node_addresses:
                self.send_block(n_addr[0], block, b_index)

    def send_block(self, outgoing_address, block, height):
        chunk_map = self.chunkify(block)
        chunks_to_send = list(chunk_map.keys())
        while chunks_to_send:
            chunk_id = random.choice(chunks_to_send)
            self.send_chunk_message(outgoing_address, chunk_map[chunk_id], height)
            chunks_to_send.remove(chunk_id)

    def chunkify(self, block):
        # This is a placeholder. You need to implement actual chunkification logic.
        return {0: Chunk(block.blockID, 0, 0, 100, 10, 10)}

    def send_chunk_message(self, outgoing_address, chunk, height):
        print(f"Sending chunk {chunk.chunkID} of block {chunk.blockID} to {outgoing_address}")

    def bucket_index_from_id(self, node_id):
        d = self.distance(self.nodeID, node_id)
        for i in range(KAD_ID_LEN):
            if (d >= 2**i) and (d < 2**(i + 1)):
                return i
        return 0

    def distance(self, node1, node2):
        return bin(node1 ^ node2).count('1')

    def handle_read(self, packet, sender_addr):
        # This is a placeholder for handling incoming packets.
        print(f"Received packet from {sender_addr}")

    def handle_sent(self, available_bytes):
        print(f"Sent {available_bytes} bytes")
        self.send_available()

    def send_available(self):
        if not self.send_queue or self.sending:
            return
        self.sending = True
        space_in_queue = 1000  # Placeholder for the actual size
        to_send = min(len(self.send_queue), space_in_queue)

        print(f"Sending {to_send} packets")
        for i in range(to_send):
            next_addr = self.send_queue.pop(0)
            self.send_packet(next_addr)

    def send_packet(self, next_addr):
        print(f"Sending packet to {next_addr}")

    def start_mining(self):
        print("Starting mining process")

# Usage example:
node = KadcastNode("192.168.1.1", True, 10.0)
node.start_application()

