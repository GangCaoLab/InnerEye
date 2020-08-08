import fire
import h5py

def ls(path: str):
    with h5py.File(path, 'r') as f:
        print(dict(f))


if __name__ == "__main__":
    fire.Fire()
