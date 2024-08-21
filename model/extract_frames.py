from tqdm import tqdm
import os
from subprocess import call
from glob import glob
import contextlib
import sys

class DummyTqdmFile(object):
    """Dummy file-like that will write to tqdm"""
    file = None
    def __init__(self, file):
        self.file = file

    def write(self, x):
        # Avoid print() second call (useless \n)
        if len(x.rstrip()) > 0:
            tqdm.write(x, file=self.file)

    def flush(self):
        return getattr(self.file, "flush", lambda: None)()

@contextlib.contextmanager
def std_out_err_redirect_tqdm():
    orig_out_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = map(DummyTqdmFile, orig_out_err)
        yield orig_out_err[0]
    # Relay exceptions
    except Exception as exc:
        raise exc
    # Always restore sys.stdout/err if necessary
    finally:
        sys.stdout, sys.stderr = orig_out_err

folders = ['UCF-101_train','UCF-101_test']
with std_out_err_redirect_tqdm() as orig_stdout:
	pbar = tqdm(total=10690, file=orig_stdout, dynamic_ncols=True)
	for folder in folders:
		class_folders = glob(os.path.join(folder, '*'))
		for vid_class in class_folders:
			class_files = glob(os.path.join(vid_class, '*.avi'))
			for vid_path in class_files:
				# print(vid_path.split('\\'))
				# print(vid_path.split('\\')[-1].split('.')[0])
				src = vid_path
				filename_no_ext = vid_path.split('\\')[-1].split('.')[0]
				dest_dir = os.path.join(vid_path.split('\\')[0]+'_seq','/'.join(vid_path.split('\\')[1:-1]),filename_no_ext)
				if(not os.path.exists(dest_dir)):
					os.makedirs(dest_dir)
				dest = os.path.join(dest_dir, filename_no_ext+'-%04d.jpg')
				# print(dest)
				call(['ffmpeg', '-i', src, dest, '-hide_banner'])
				pbar.update(1)