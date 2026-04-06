import json
import argparse

def convert(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    flat_entries = {}  # will become a dict mapping image_id -> entry dict
    for cam_key, cam_info in data.items():
        common = {}
        image_entries = {}
        for key, value in cam_info.items():
            if isinstance(value, dict) and 'file_path' in value:
                image_entries[key] = value
            else:
                common[key] = value
        # add cam_key as a separate field if needed, but do not include it as "group"
        common["cam_id"] = cam_key

        if image_entries:
            for img_key, img_info in image_entries.items():
                entry = common.copy()
                entry.update(img_info)

                flat_entries[img_key] = entry
        else:
            print(f"WARNING: No entries found cam_id {cam_key}")

    with open(output_path, 'w') as f:
        json.dump(flat_entries, f, indent=2)
    print(f"Converted file saved to {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Convert hierarchical transformation json to flat json format.')
    parser.add_argument('--input', type=str, required=True,
                        help='Path to the input hierarchical json file')
    parser.add_argument('--output', type=str, required=True,
                        help='Path to the output flat json file')
    args = parser.parse_args()
    convert(args.input, args.output)
