# The parser here is not designed to be a well-coded library with
# good performance, it is more like a demo for showing the data
# structure.
import os
import os.path
import math
import zlib
import struct
import hashlib
import shutil

FILENAME_MASK1 = "olhwjsktri"
FILENAME_MASK2 = "eizxdwknmo"
FILENAME_ENCODING = {k: v for v, k in enumerate(FILENAME_MASK1)}

MAP_WIDTH = 512
TILE_WIDTH = 128
TILE_HEADER_LEN = TILE_WIDTH**2
TILE_HEADER_SIZE = TILE_HEADER_LEN * 2
BLOCK_BITMAP_SIZE = 512
BLOCK_EXTRA_DATA = 3
BLOCK_SIZE = BLOCK_BITMAP_SIZE + BLOCK_EXTRA_DATA
BITMAP_WIDTH = 64

NNZ_FOR_BYTE = bytes(bin(x).count("1") for x in range(256))


def nnz(data):
    """count number of bit 1s in given piece of data."""
    count = 0
    for b in data:
        count = count + NNZ_FOR_BYTE[b]
    return count


class Block:
    def __init__(self, x, y, data):
        self.x = x
        self.y = y
        self.bitmap = bytearray(data[:BLOCK_BITMAP_SIZE])
        self.extra_data = bytearray(data[BLOCK_BITMAP_SIZE:BLOCK_SIZE])
        # extra_data has three bytes, the bitwise representation is: XXXX XYYY YY0Z ZZZZ ZZZZ ZZZ1, where:
        # XXXXX: first char of region string, offset by ascii "?"
        # YYYYY: second char of region string, offset by ascii "?"
        # ZZZZZZZZZZZZ: number of 1s in bitmap, from 1 to 4196
        region_str0 = ((self.extra_data[0] >> 3) + b"?"[0]).to_bytes(1, "big").decode()
        region_str1 = (
            (
                (((self.extra_data[0] & 0x7) << 2) | ((self.extra_data[1] & 0xC0) >> 6))
                + b"?"[0]
            )
            .to_bytes(1, "big")
            .decode()
        )
        self.region = region_str0 + region_str1
        if self.region == "@@":
            self.region = "BORDER/INTERNATIONAL"
        assert self.validate_checksum()

    def validate_checksum(self):
        checksum = struct.unpack(">H", self.extra_data[1:])[0] & 0x3FFF
        valid = nnz(self.bitmap) << 1 == checksum - 1
        if not valid:
            print(
                "WARNING: block ({},{}) checksum is not correct.".format(self.x, self.y)
            )
        return valid

    def update_extra_data(self):
        extra_data = bytearray(3)
        region_str0_b = ord(self.region[0]) - ord("?")
        region_str1_b = ord(self.region[1]) - ord("?")
        extra_data[0] = (region_str0_b << 3) | (region_str1_b >> 2)
        checksum = nnz(self.bitmap)
        extra_data[1] = (region_str1_b & 0x3) << 6 | checksum >> 7
        extra_data[2] = ((checksum & 0x7F) << 1) | 1
        self.extra_data = extra_data
        assert self.validate_checksum()

    def is_visited(self, x, y):
        bit_offset = 7 - x % 8
        i = x // 8
        j = y
        return self.bitmap[i + j * 8] & (1 << bit_offset)

    def to_bytearray(self):
        return self.bitmap + self.extra_data

    def __eq__(self, other):
        assert self.x == other.x and self.y == other.y and self.region == other.region
        return self.bitmap == other.bitmap

    def from_template(other, bitmap):
        other.update_extra_data()
        new_block = other
        other.bitmap = bitmap
        new_block.update_extra_data()

        return new_block

    def __sub__(self, other: "Block") -> "Block":
        assert self.x == other.x and self.y == other.y
        assert self.region == other.region
        bitmap = bytearray(
            [self.bitmap[i] & ~other.bitmap[i] for i in range(len(self.bitmap))]
        )
        return Block.from_template(self, bitmap)


def _tile_x_y_to_lng_lat(x: int, y: int):
    lng = x / 512 * 360 - 180
    lat = math.atan(math.sinh(math.pi - 2 * math.pi * y / 512)) * 180 / math.pi
    return (lng, lat)


class Tile:
    def __init__(self, sync_folder, filename):
        self.filename = filename
        self.file = os.path.join(sync_folder, filename)
        # parse filename
        self.id = 0
        for v in [FILENAME_ENCODING[c] for c in filename[4:-2]]:
            self.id = self.id * 10 + v
        self.x = self.id % MAP_WIDTH
        self.y = self.id // MAP_WIDTH
        print("Loading tile. id: {}, x: {}, y: {}".format(self.id, self.x, self.y))

        # filename should start with md5(tileId) and end with mask2(tileId[-2:])
        match1 = hashlib.md5(str(self.id).encode()).hexdigest()[0:4] == filename[0:4]
        match2 = (
            "".join([FILENAME_MASK2[int(i)] for i in str(self.id)[-2:]])
            == filename[-2:]
        )
        if not (match1 and match2):
            print("WARNING: the filename {} is not valid.".format(filename))

        with open(self.file, "rb") as f:
            data = f.read()
            data = zlib.decompress(data)
        # header is a 2d array of shorts, it contains the maping of blocks
        header = struct.unpack(str(TILE_HEADER_LEN) + "H", data[:TILE_HEADER_SIZE])
        self.blocks = {}
        self.region_set = set()
        for i, block_idx in enumerate(header):
            if block_idx > 0:
                block_x = i % TILE_WIDTH
                block_y = i // TILE_WIDTH
                start_offset = TILE_HEADER_SIZE + (block_idx - 1) * BLOCK_SIZE
                end_offset = start_offset + BLOCK_SIZE
                block_data = data[start_offset:end_offset]
                block = Block(block_x, block_y, block_data)
                self.region_set.add(block.region)
                self.blocks[(block_x, block_y)] = block

    def bounds(self):
        (lng1, lat1) = _tile_x_y_to_lng_lat(self.x, self.y)
        (lng2, lat2) = _tile_x_y_to_lng_lat(self.x + 1, self.y + 1)
        return ((min(lat1, lat2), min(lng1, lng2)), (max(lat1, lat2), max(lng1, lng2)))

    def dump(self, outdir=None):
        block_idx = 0
        data = []
        header = [0 for i in range(TILE_HEADER_LEN * 2)]
        for y in range(TILE_WIDTH):
            for x in range(TILE_WIDTH):
                if (x, y) in self.blocks:
                    block_idx += 1
                    block = self.blocks[(x, y)]
                    data += block.to_bytearray()
                    idx = y * TILE_WIDTH + x
                    header[idx * 2 + 1] = block_idx >> 8
                    header[idx * 2] = block_idx & 0xFF
        data = zlib.compress(bytes(header + data))
        p = os.path.join(outdir, self.filename) if outdir else self.file
        with open(p, "wb") as f:
            f.write(data)

    def from_template(other, blocks, region_set):
        new_tile = other
        new_tile.blocks = blocks
        new_tile.region_set = region_set
        return new_tile

    def __eq__(self, other: "Tile") -> bool:
        return self.blocks == other.blocks

    def __sub__(self, other: "Tile"):
        blocks = {}
        region_set = set()
        for k, v in self.blocks.items():
            if k not in other.blocks:
                blocks[k] = v
                region_set.add(blocks[k].region)
                continue
            if v == other.blocks[k]:
                continue
            blocks[k] = v - other.blocks[k]
            region_set.add(blocks[k].region)
        return Tile.from_template(self, blocks, region_set)


class FogMap:
    # The toplevel class that represent the whole data.
    # The whole map is divided into 512*512 tiles. Each tile
    # is a file and it includes 128*128 blocks. Each block is
    # a 64*64 bitmap.
    def from_dir(path):
        assert os.path.isdir(path)
        tile_map = {}
        region_set = set()
        for filename in os.listdir(path):
            tile = Tile(path, filename)
            region_set.update(tile.region_set)
            tile_map[(tile.x, tile.y)] = tile
        print("Traversed region: {}".format(region_set))
        return FogMap(tile_map, region_set)

    def __init__(self, tile_map, region_set):
        self.tile_map = tile_map
        self.region_set = region_set

    def __sub__(self, other: "FogMap") -> "FogMap":
        tile_map = {}
        region_set = set()

        for k, v in self.tile_map.items():
            if k not in other.tile_map:
                tile_map[k] = v
                region_set.update(tile_map[k].region_set)
                continue
            if v == other.tile_map[k]:
                continue
            tile_map[k] = v - other.tile_map[k]
            region_set.update(tile_map[k].region_set)
        print("New traversed region: {}".format(self.region_set))
        return FogMap(tile_map, region_set)

    def dump(self, outdir="output"):
        if os.path.exists(outdir):
            print(f"{outdir} already exists, removing.")
            shutil.rmtree(outdir)
        os.mkdir(outdir)
        for _, v in self.tile_map.items():
            v.dump(outdir)


def sub_map():
    fog_map = FogMap.from_dir("20231010.bitmap")
    google_map = FogMap.from_dir("202305xx.raw.bitmap")

    new_map = fog_map - google_map
    new_map.dump()


def sub_tile():
    oldtile = Tile("202305xx.raw.bitmap", "fb6floojwkxk")
    newtile = Tile("20231010.bitmap", "fb6floojwkxk")
    t = newtile - oldtile
    t.dump("tmp")


sub_map()
