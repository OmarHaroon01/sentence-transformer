import os
import requests
from kaggle.api.kaggle_api_extended import KaggleApi
from tqdm import tqdm

class MyKaggleApi(KaggleApi):
    def kernels_output(self, kernel, path, force=False, quiet=True):
            """ retrieve output for a specified kernel
                Parameters
                ==========
                kernel: the kernel to output
                path: the path to pull files to on the filesystem
                force: if output already exists, force overwrite (default False)
                quiet: suppress verbosity (default is True)
            """
            if kernel is None:
                raise ValueError('A kernel must be specified')
            if '/' in kernel:
                self.validate_kernel_string(kernel)
                kernel_url_list = kernel.split('/')
                owner_slug = kernel_url_list[0]
                kernel_slug = kernel_url_list[1]
            else:
                owner_slug = self.get_config_value(self.CONFIG_NAME_USER)
                kernel_slug = kernel

            if path is None:
                target_dir = self.get_default_download_dir('kernels', owner_slug,
                                                        kernel_slug, 'output')
            else:
                target_dir = path

            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            if not os.path.isdir(target_dir):
                raise ValueError(
                    'You must specify a directory for the kernels output')

            response = self.process_response(
                self.kernel_output_with_http_info(owner_slug, kernel_slug))
            outfiles = []
            for item in response['files']:
                outfile = os.path.join(target_dir, item['fileName'])
                outfiles.append(outfile)
                download_response = requests.get(item['url'])
                if force or self.download_needed(item, outfile, quiet):
                    os.makedirs(os.path.split(outfile)[0], exist_ok=True)
                    with open(outfile, 'wb') as out:
                        out.write(download_response.content)
                    if not quiet:
                        print('Output file downloaded to %s' % outfile)


def download_kaggle_kernel_output(kernel_slug, path='sonar/'):
    api = MyKaggleApi()
    api.authenticate()

    # Ensure the path exists
    if not os.path.exists(path):
        os.makedirs(path)

    # List output files for the specified kernel
    output_files = api.kernels_output(kernel_slug, path)



if __name__ == "__main__":

    
    file_names = [
        'lamia1ieeeacc/s-0-20000',
        'merliahsummer/s-20k-50k',
        'nasreeeen/s-50k-90k',
        'mohsinamoumi/0-9-1-2',
        'moumi16/1-2-1-5',
        'googlecollab1/1-5-2-0',
        'googlecollab2/2-0-2-5',
        'narubean/2-5-3-0',
        'lamiamuni/s-3-0-3-5',
        'maishaamin2013788642/s-3-5-4',
        'sarailli/s-4-4-5',
        'zarasara/s-4-5-5',
        'narubean/5-0-5-5',
        'mohsinamatinmoumi/5-5-6-0',
        'shahanamahmud/6-0-6-5',
        'aaronbagerrahman/6-5-7-0',
        'hatirjheel/7-0-7-5',
        'mohsinamoumi/7-5-8-0',
        'moumi16/8-0-8-5',
        'googlecollab1/8-5-9-0',
        'mehrinmatinmomo/9-0-9-5',
        'googlecollab2/9-5-10-0',
        'lamiamunira/s-10-11',
        'merliahsummer/s-11-12',
        'nasreeeen/s-12-13',
        'zsultana/s-13-14',
        'lamia1ieeeacc/s-14-15',
        'narubean/15-0-16-0',
        'mohsinamatinmoumi/16-0-17-0',
        'shahanamahmud/17-0-18-0',
        'aaronbagerrahman/18-0-19-0',
        'mehrinmatinmomo/19-0-20-0',
        'hatirjheel/20-0-21-0',
        'omarharoon/21-22',
        'omarharoon01/22-23',
        'bilkisakter/23-24',
        'nabihabharoon/24-25',
        'maishaamin2013788642/s-25-26',
        'sarailli/s-26-27',
        'zarasara/s-27-28',
        'lamia1ieeeacc/s-28-29',
        'lamiamuni/s-29-30',
        'narubean/30-1-31-0',
        'googlecollab1/31-0-32-0',
        'googlecollab2/32-0-33-0',
        'moumi16/33-0-34-0',
        'mohsinamoumi/34-0-35-0',
        'hatirjheel/35-0-36-0',
        'mohsinamatinmoumi/36-0-37-0',
        'shahanamahmud/37-0-38-0',
        'aaronbagerrahman/38-0-39-0',
        'mehrinmatinmomo/notebook5ab815c23b',
        'lamiamunira/s-40-41',
        'lamiamuni/s-41-42',
        'merliahsummer/s-42-43',
        'zsultana/s-43-44',
        'nasreeeen/s-44-45',
        'lamiamunira/s-45-46',
        'lamiamuni/s-46-47',
        'omarharoon/s47-48',
        'bilkisakter/s48-49',
        'nasreeeen/s-49-50',
        'hatirjheel/50-0-51-0',
        'mohsinamatinmoumi/51-0-52-0',
        'shahanamahmud/52-0-53-0',
        'mehrinmatinmomo/53-0-54-0',
        'aaronbagerrahman/54-0-55-0',
        'omarharoon/55-0-56-0',
        'omarharoon01/56-57',
        'bilkisakter/57-58',
        'nabihabharoon/58-59',
        'omarharoon/59-60',
        'omarharoon01/60-61-5',
        'bilkisakter/61-5-63',
        'nabihabharoon/63-65'
    ]
    
    cnt = 0
    print(f'Total file count is {len(file_names)}')
    for file in file_names:
        download_kaggle_kernel_output(file)
        cnt += 1
        print(f'File {cnt} done')
