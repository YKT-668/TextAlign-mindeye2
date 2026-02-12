import h5py
p="/mnt/work/repos/TextAlign-mindeye2/COCO_73k_subj_indices.hdf5"
print("PATH:", p)
f=h5py.File(p,"r")
def walk(name, obj):
    if hasattr(obj, "shape"):
        print(name, "shape=", obj.shape, "dtype=", obj.dtype)
f.visititems(walk)
f.close()
