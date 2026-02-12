import sys, os
import h5py
import numpy as np

def main(p: str):
    print("PATH:", p)
    print("exists?", os.path.exists(p))
    if not os.path.exists(p):
        return 1

    with h5py.File(p, "r") as f:
        keys = list(f.keys())
        print("keys:", keys)

        # 1D int & very long: typically id mapping
        cands = []
        for k in keys:
            ds = f[k]
            if hasattr(ds, "shape") and len(ds.shape) == 1 and ds.shape[0] > 70000:
                try:
                    x = ds[:5]
                    if np.issubdtype(x.dtype, np.integer):
                        cands.append((k, int(ds.shape[0]), str(x.dtype), x.tolist()))
                except Exception:
                    pass

        print("\n1D int datasets (len>70000):")
        if not cands:
            print("  (none)")
        else:
            for k, n, dt, head in cands:
                print(f" - {k}: len={n} dtype={dt} head={head}")

        # 4D dataset: usually images
        print("\n4D datasets:")
        any4d = False
        for k in keys:
            ds = f[k]
            if hasattr(ds, "shape") and len(ds.shape) == 4:
                any4d = True
                print(f" - {k}: shape={ds.shape} dtype={ds.dtype}")
        if not any4d:
            print("  (none)")

    return 0

if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "/mnt/work/repos/TextAlign-mindeye2/src/coco_images_224_float16.hdf5"
    raise SystemExit(main(p))