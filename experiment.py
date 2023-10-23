from glob import glob
import os


def main():
    output_dir = "result"
    dataset_dir = "dataset"
    types = ("*.bmp", "*.png")

    files = []
    for filetype in types:
        files.extend(glob(f'{dataset_dir}/**/{filetype}'))

    blurring_techniques = ['gaussian', 'median']
    kernel_size = 7

    for blur in blurring_techniques:
        for file in files:
            command = f"python main.py --blur {blur} -k {kernel_size} -v -o {output_dir}/{blur} {file}"
            print(command)
            os.system(command)
        
if __name__ == '__main__':
    main()
