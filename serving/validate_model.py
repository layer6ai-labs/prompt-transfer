"""Check a downloaded Hugging Face safetensors checkpoint (stdlib only)."""

import argparse
import json
from pathlib import Path
import struct


def validate_shard(path):
    """Check header/payload extents without loading the tensor data into RAM.

    A nonempty shard may still be a partial download. Safetensors records each
    tensor's byte range in its JSON header, so the final range must end exactly
    at the file's payload size. This is a completeness check, not a checksum.
    """
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"{path.name}: missing safetensors header")
        header_size, = struct.unpack("<Q", prefix)
        if not 0 < header_size <= size - 8:
            raise ValueError(f"{path.name}: truncated safetensors header")
        header = json.loads(stream.read(header_size))
    if not isinstance(header, dict):
        raise ValueError(f"{path.name}: safetensors header must be an object")
    ranges = []
    for name, tensor in header.items():
        if name == "__metadata__":
            continue
        offsets = tensor.get("data_offsets") if isinstance(tensor, dict) else None
        if (not isinstance(offsets, list) or len(offsets) != 2
                or any(type(value) is not int for value in offsets)
                or not 0 <= offsets[0] <= offsets[1]):
            raise ValueError(f"{path.name}: invalid tensor offsets for {name}")
        ranges.append(offsets)
    if not ranges:
        raise ValueError(f"{path.name}: safetensors header has no tensors")
    end = 0
    for start, stop in sorted(ranges):
        if start != end:
            raise ValueError(f"{path.name}: tensor payload has gaps or overlaps")
        end = stop
    if end != size - 8 - header_size:
        raise ValueError(f"{path.name}: truncated or oversized tensor payload")
    return size


def validate_model(directory):
    directory = Path(directory)
    config = directory / "config.json"
    if not config.is_file() or not config.stat().st_size:
        raise ValueError("missing or empty config.json")
    if not isinstance(json.loads(config.read_text()), dict):
        raise ValueError("config.json must contain an object")

    shards = {path.name for path in directory.glob("*.safetensors") if path.is_file()}
    index = directory / "model.safetensors.index.json"
    if index.exists():
        if not index.is_file() or not index.stat().st_size:
            raise ValueError("empty safetensors index")
        contents = json.loads(index.read_text())
        weight_map = contents.get("weight_map") if isinstance(contents, dict) else None
        if not isinstance(weight_map, dict) or not weight_map:
            raise ValueError("safetensors index has no weights")
        needed = set(weight_map.values())
        if any(not isinstance(name, str) or Path(name).name != name or not name.endswith(".safetensors")
               for name in needed):
            raise ValueError("safetensors index contains an invalid shard filename")
    else:
        # Single-file HF checkpoints do not have an index. A set of sharded
        # files without its index cannot establish which weights are required.
        needed = {"model.safetensors"}
    if not shards or shards != needed:
        raise ValueError(f"incomplete shard set: missing={sorted(needed - shards)}, extra={sorted(shards - needed)}")
    return len(needed), sum(validate_shard(directory / name) for name in sorted(needed))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directories", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for directory in args.directories:
        try:
            count, size = validate_model(directory)
        except (OSError, ValueError, TypeError) as exc:
            print(f"{directory.name}: INCOMPLETE: {exc}")
            failed = True
        else:
            print(f"{directory.name}: {count} complete shards, {size / 1e9:.1f} GB, ok")
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
